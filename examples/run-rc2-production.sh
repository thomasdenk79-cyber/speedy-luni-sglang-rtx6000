#!/usr/bin/env bash
# rc2: the exact production configuration of qwen38-rc2 (1x RTX PRO 6000
# Blackwell 96 GB, SM120), in service since 2026-11-18.
#
# Stock SGLang nightly image + the rc2 source tree bind-mounted to /src
# (PYTHONPATH=/src/python), so no image build is needed.
#
#   git clone -b speedy-rc2 https://github.com/thomasdenk79-cyber/sglang "$SRC_DIR"
#   git clone https://github.com/thomasdenk79-cyber/speedy-luni-sglang-rtx6000
#   MODEL_DIR=/path/to/Qwen3.8-Flash-Next-FP4 ./examples/run-rc2-production.sh
#
# DRY_RUN=1 prints the command and exits. Host needs: NVIDIA driver with CDI
# (podman: --device nvidia.com/gpu=all; docker: GPU_ARGS="--gpus all"),
# ~40 GB free host RAM (pinned PLE table + HiCache buffer), and a fast NVMe for
# the HiCache file tier (HICACHE_MAX, default 200G).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-docker.io/lmsysorg/sglang:nightly-dev-cu13-20261117-25c1c439}"
CONTAINER_CLI="${CONTAINER_CLI:-podman}"
GPU_ARGS="${GPU_ARGS:---device nvidia.com/gpu=all}"
SRC_DIR="${SRC_DIR:-/srv/community-repos/sglang-wt-rc2}"
MODEL_DIR="${MODEL_DIR:-/srv/llm-models/Qwen3.8-Flash-Next-FP4-Local}"
CACHE_DIR="${CACHE_DIR:-/srv/llm-cache/qwen38-flashnext}"
# Production NVMe prefix cache; a different directory starts with an empty cache.
HICACHE_DIR="${HICACHE_DIR:-$CACHE_DIR/hicache/file-nvfp4-9070b9dca5f3}"
HICACHE_MAX="${HICACHE_MAX:-200G}"
ADAPTIVE_CONFIG="${ADAPTIVE_CONFIG:-$HERE/configs/adaptive-next-tiered.json}"
TOKEN_MAP="${TOKEN_MAP:-$HERE/tokenmaps/hyudryu_top49152.pt}"
PORT="${PORT:-30000}"
NAME="${NAME:-qwen38-rc2}"
MAX_RUNNING="${MAX_RUNNING:-8}"
# No default on purpose (Betreiber 22.11.): SGLang selbst bestimmt den Pool aus
# dem freien VRAM. --max-total-tokens nur in begruendeten A/B-Tests per Env
# setzen (MAX_TOTAL_TOKENS=... ./run-rc2-production.sh), nie als Produktion-Default.
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
MEM_FRACTION="${MEM_FRACTION:-0.96}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-524288}"
DRY_RUN="${DRY_RUN:-0}"

for p in "$SRC_DIR/python/sglang" "$MODEL_DIR" "$ADAPTIVE_CONFIG" "$TOKEN_MAP"; do
  [[ -e $p || $DRY_RUN == 1 ]] || { echo "missing: $p" >&2; exit 2; }
done
[[ $DRY_RUN == 1 ]] || mkdir -p "$CACHE_DIR" "$HICACHE_DIR"

C=/var/cache/qwen38
ENVS=(
  PYTHONPATH=/src/python
  HF_HOME=$C/huggingface SGLANG_CACHE_DIR=$C/sglang SGLANG_JIT_CACHE_DIR=$C/sglang/jit
  CUTE_DSL_CACHE_DIR=$C/sglang/cute_dsl CUDA_CACHE_PATH=$C/cuda TRITON_CACHE_DIR=$C/triton
  TORCHINDUCTOR_CACHE_DIR=$C/torchinductor
  TORCH_EXTENSIONS_DIR=$C/sglang/torch_extensions/py312_torch2.14.1+cu130
  CARGO_HOME=$C/rust/cargo RUSTUP_HOME=$C/rust/rustup RUSTUP_AUTO_INSTALL=0
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
  SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1
  SGLANG_SM120_ONLINE_MXFP8=true SGLANG_SM120_BF16_SKINNY_GEMV=1 SGLANG_OPT_HC_GATE_STREAM_SM120=0
  FLASHINFER_GDN_FP16_ACCUM_MMA=1
  SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=true SGLANG_MAMBA_CONV_DTYPE=bfloat16
  SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=rust
  SGLANG_STARTUP_AUTOTUNE_REPLAY=1 SGLANG_STARTUP_CSF_FAST_DECODE=1
  SGLANG_DRAFT_SKIP_VOCAB_WEIGHTS=1 SGLANG_SPEC_DYNAMIC_VOCAB_SLOTS=8192
  SGLANG_HICACHE_BUFFER_STREAM=1
  SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/hicache
  SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=$HICACHE_MAX
  SGLANG_HICACHE_FILE_BACKEND_READ_WORKERS=16
  SGLANG_HICACHE_FILE_BACKEND_ZERO_COPY_READ=1
)
ENV_ARGS=(); for e in "${ENVS[@]}"; do ENV_ARGS+=(-e "$e"); done

ROPE='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}'

ARGS=(
  --model-path /models/qwen38 --host 0.0.0.0 --port "$PORT" --enable-metrics
  --dtype bfloat16 --quantization modelopt_fp4 --moe-runner-backend flashinfer_cutlass_sm120_swiglu
  --weight-loader-prefetch-checkpoints --weight-loader-prefetch-num-threads 4 --weight-loader-drop-cache-after-load
  --kv-cache-dtype nvfp4 --mem-fraction-static "$MEM_FRACTION" --page-size 64
  --context-length "$CONTEXT_LENGTH" --json-model-override-args "$ROPE"
  --max-running-requests "$MAX_RUNNING" --chunked-prefill-size 8192 --prefill-decode-interval 1
  --mamba-radix-cache-strategy extra_buffer_lazy --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 48
  --hybrid-cache-checkpoints-level 2 --mamba-track-interval 64
  --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer
  --cuda-graph-bs-decode 1 2 3 4 5 6 8 12 16 --cuda-graph-max-bs-prefill 1024
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
  --enable-linear-replayssm-spec --ple-offload-embedding
  --speculative-algorithm NEXTN --speculative-eagle-topk 1 --speculative-num-steps 7 --speculative-num-draft-tokens 8
  --speculative-draft-kv-cache-dtype fp8_e4m3 --speculative-draft-model-quantization unquant
  --speculative-adaptive --speculative-adaptive-config /config/adaptive.json --speculative-token-map /config/token_map.pt
  --enable-hierarchical-cache --hicache-host-memory-mode buffer_only --hicache-size 4
  --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first
  --hicache-storage-backend file --hicache-storage-prefetch-policy timeout
)
# Max-Total-Tokens nur bei ausdruecklicher Setzung (A/B-Tests) — sonst
# bestimmt SGLang die Poolgroesse aus dem verfuegbaren VRAM.
if [[ -n $MAX_TOTAL_TOKENS ]]; then ARGS+=(--max-total-tokens "$MAX_TOTAL_TOKENS"); fi

CMD=("$CONTAINER_CLI" run -d --name "$NAME" --restart unless-stopped $GPU_ARGS
  --network host --ipc host --ulimit memlock=-1:-1 --user 0:0
  -v "$SRC_DIR:/src:ro" -v "$MODEL_DIR:/models/qwen38:ro"
  -v "$ADAPTIVE_CONFIG:/config/adaptive.json:ro" -v "$TOKEN_MAP:/config/token_map.pt:ro"
  -v "$CACHE_DIR:$C" -v "$HICACHE_DIR:/hicache"
  "${ENV_ARGS[@]}" --entrypoint python3 "$IMAGE" -m sglang.launch_server "${ARGS[@]}")

if [[ $DRY_RUN == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
"${CMD[@]}"
echo "started $NAME; first start JIT-compiles kernels (several minutes)."
echo "health: curl -s http://127.0.0.1:$PORT/health ; logs: $CONTAINER_CLI logs -f $NAME"
