#!/usr/bin/env python3
"""Compare the NVFP4-KV QSA attention paths on one card, one shape, one truth.

Families (``--families``):

``ours_bf16``
    Our Triton gather/dequant (``sglang.srt.layers.attention.qsa.nvfp4_kv``) into a
    bf16 scratch, then ordinary attention - what ``rc/qwen38-rc2`` runs today.
``fi_ref``
    ``flashinfer.sparse.qsa_sparse_paged_attention(..., backend="default")``: the
    reference build, SM80/SM90 by design.
``fi_cake``
    The same entry with ``backend="cake"``: the Cake kernels, which ``backend="auto"``
    selects on SM120 (``(12, 0)`` in ``cake_jit.ARCH_BY_CAPABILITY``).
``reference``
    Pure-torch exact attention over :func:`reference_dequant` output.  Always runs and
    is the truth the other families are scored against.

The physical KV pool is built once as packed E2M1 plus E4M3 block scales, so every
family sees bit-identical bytes; only the kernel differs.  A family that cannot load
is recorded as ``unavailable``/``error`` with its message - it is never scored as slow.

Contract sources (verified 2026-11-22 in ``lmsysorg/sglang:nightly-dev-cu13-20261117-25c1c439``):
``flashinfer.sparse.qsa_sparse_paged_attention(q, k_cache, v_cache, logical_indices,
block_table, token_to_req, use_prefill_config, out=None, k_scale=None, v_scale=None, *,
output_gate, k_sf=None, v_sf=None, backend="auto")`` and the module constants
``HEAD_DIM=256``, ``TILE_TOKENS=64``, ``MAX_SPLITS=128`` of
``flashinfer.experimental.cake_qsa_nvfp4.cake_backend``.

Examples
    # CPU smoke test of the plumbing (reference only, no CUDA needed)
    python3 tools/nvfp4_kernel_compare.py --device cpu --rows 4 --width 128

    # the GPU comparison inside the owner window
    python3 tools/nvfp4_kernel_compare.py --repo /work --families ours_bf16,fi_ref,fi_cake \\
        --rows 1 4 16 64 --width 2048 --reps 7 --l2-flush both --capture \\
        --json-out /work/kv-persist/nvfp4-compare.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

# Qwen3.8-Flash-Next QSA layer at TP1, as pinned by
# test/registered/kernels/benchmark/attention/bench_qsa_sparse_decode.py
QO_HEADS, KV_HEADS, HEAD_DIM = 24, 2, 256
PACKED_DIM = HEAD_DIM // 2          # 128 bytes of packed E2M1 per head
SF_PER_HEAD = HEAD_DIM // 16         # 16 E4M3 block scales per head
PAGE = 64                            # cache page in tokens
TILE_TOKENS = 64                     # flashinfer cake tile, verified constant
E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

FAMILIES = ("ours_bf16", "fi_ref", "fi_cake")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NVFP4-KV attention path comparison")
    p.add_argument("--repo", help="sglang checkout to import our kernels from")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--families", default=",".join(FAMILIES))
    p.add_argument("--rows", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--width", type=int, default=2048, help="selected tokens per row")
    p.add_argument("--context", type=int, default=0, help="tokens per request (default 2*width)")
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--l2-flush", default="both", choices=("on", "off", "both"))
    p.add_argument("--capture", action="store_true", help="also time a CUDA-graph replay")
    p.add_argument("--sm-scale", type=float, default=None, help="default 1/sqrt(head_dim)")
    p.add_argument("--k-scale", type=float, default=0.01)
    p.add_argument("--v-scale", type=float, default=0.013)
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--json-out", help="write the result rows here (JSONL)")
    p.add_argument("--fail-fast", action="store_true", help="stop at the first family error")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def build_case(args: argparse.Namespace, rows: int, device: torch.device) -> dict:
    """One deterministic case: packed NVFP4 pool, selection, queries, gate."""
    gen = torch.Generator(device="cpu").manual_seed(args.seed + rows)
    context = args.context or 2 * args.width
    if args.width > context:
        raise SystemExit("--width must not exceed --context")
    pages_per_req = -(-context // PAGE)
    num_requests = rows                      # one decode row per request
    num_blocks = num_requests * pages_per_req + 1

    def randint(*shape: int, high: int) -> torch.Tensor:
        return torch.randint(high, shape, generator=gen, dtype=torch.int64)

    data_k = randint(num_blocks, PAGE, KV_HEADS, PACKED_DIM, high=256).to(torch.uint8)
    data_v = randint(num_blocks, PAGE, KV_HEADS, PACKED_DIM, high=256).to(torch.uint8)
    # E4M3 bytes: keep the finite positive range (0x00..0x7E) so scales stay usable
    sf_k = randint(num_blocks, PAGE, KV_HEADS, SF_PER_HEAD, high=0x7F).to(torch.uint8)
    sf_v = randint(num_blocks, PAGE, KV_HEADS, SF_PER_HEAD, high=0x7F).to(torch.uint8)

    block_table = torch.empty(num_requests, pages_per_req, dtype=torch.int32)
    for req in range(num_requests):
        block_table[req] = torch.randperm(num_blocks - 1, generator=gen)[:pages_per_req] + 1
    block_table = block_table.to(device)

    # every row selects `width` distinct logical tokens out of its context
    logical = torch.stack([torch.randperm(context, generator=gen)[: args.width]
                           for _ in range(rows)]).sort(dim=1).values
    logical_indices = torch.empty(rows, args.width + 1, dtype=torch.int32)
    logical_indices[:, : args.width] = logical
    logical_indices[:, args.width] = args.width          # valid count, last column
    logical_indices = logical_indices.to(device)

    q = torch.randn(rows, QO_HEADS, HEAD_DIM, generator=gen, dtype=torch.float32).bfloat16()
    gate = torch.randn(rows, QO_HEADS, HEAD_DIM, generator=gen, dtype=torch.float32).bfloat16()
    tokens = {
        "data_k": data_k.to(device), "data_v": data_v.to(device),
        "sf_k": sf_k.to(device).view(torch.float8_e4m3fn),
        "sf_v": sf_v.to(device).view(torch.float8_e4m3fn),
        "q": q.to(device), "gate": gate.to(device),
        "logical_indices": logical_indices, "block_table": block_table,
        "token_to_req": torch.arange(num_requests, dtype=torch.int32, device=device),
        "slots": logical_to_slots(logical, block_table, num_requests),
        "context": context, "pages_per_req": pages_per_req,
    }
    return tokens


def logical_to_slots(logical: torch.Tensor, block_table: torch.Tensor,
                     num_requests: int) -> torch.Tensor:
    """Flat pool slot per selected logical token (``block * PAGE + offset``)."""
    rows, width = logical.shape
    req = torch.arange(num_requests)[:rows, None].expand(rows, width)
    page = block_table[req.long(), logical.long() // PAGE]
    return (page * PAGE + logical % PAGE).reshape(-1).to(torch.int64)


# --------------------------------------------------------------------------- #
# truth
# --------------------------------------------------------------------------- #
def reference_attention(tokens: dict, args: argparse.Namespace,
                        device: torch.device) -> torch.Tensor:
    """Exact attention over the dequantised selection: the score every family must match."""
    from sglang.srt.layers.attention.qsa.nvfp4_kv import reference_dequant

    data_k = tokens["data_k"].view(torch.uint8)
    data_v = tokens["data_v"].view(torch.uint8)
    pool_k = data_k.reshape(-1, KV_HEADS, PACKED_DIM)
    pool_v = data_v.reshape(-1, KV_HEADS, PACKED_DIM)
    sf_k = tokens["sf_k"].view(torch.uint8).reshape(-1, KV_HEADS, SF_PER_HEAD)
    sf_v = tokens["sf_v"].view(torch.uint8).reshape(-1, KV_HEADS, SF_PER_HEAD)
    ones = torch.ones((), dtype=torch.float32, device=device)
    k_all = reference_dequant(pool_k, sf_k, ones, torch.float32)
    v_all = reference_dequant(pool_v, sf_v, ones, torch.float32)

    rows, width = tokens["logical_indices"].shape[0], args.width
    slots = tokens["slots"]
    k = k_all.index_select(0, slots).view(rows, width, KV_HEADS, HEAD_DIM)
    v = v_all.index_select(0, slots).view(rows, width, KV_HEADS, HEAD_DIM)
    k = k * args.k_scale
    v = v * args.v_scale

    q = tokens["q"].float()
    groups = QO_HEADS // KV_HEADS
    k = k.repeat_interleave(groups, dim=2)
    v = v.repeat_interleave(groups, dim=2)
    sm_scale = args.sm_scale or 1.0 / (HEAD_DIM ** 0.5)
    scores = torch.einsum("rqd,rwqd->rqw", q, k) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("rqw,rwqd->rqd", probs, v)
    return (out.bfloat16() * torch.sigmoid(tokens["gate"].float()).bfloat16())


# --------------------------------------------------------------------------- #
# families
# --------------------------------------------------------------------------- #
def run_ours_bf16(tokens: dict, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    """Gather+dequant to bf16 scratch (our Triton kernel), then attention."""
    from sglang.srt.layers.attention.qsa.nvfp4_kv import nvfp4_gather_dequant

    rows = tokens["logical_indices"].shape[0]
    slots = tokens["slots"]
    pool_k = tokens["data_k"].view(torch.uint8).reshape(-1, KV_HEADS, PACKED_DIM)
    pool_v = tokens["data_v"].view(torch.uint8).reshape(-1, KV_HEADS, PACKED_DIM)
    sf_k = tokens["sf_k"].view(torch.uint8).reshape(-1, KV_HEADS, SF_PER_HEAD)
    sf_v = tokens["sf_v"].view(torch.uint8).reshape(-1, KV_HEADS, SF_PER_HEAD)
    g_k = torch.full((), args.k_scale, dtype=torch.float32, device=device)
    g_v = torch.full((), args.v_scale, dtype=torch.float32, device=device)
    k = nvfp4_gather_dequant(pool_k, sf_k, g_k, slots, torch.bfloat16)
    v = nvfp4_gather_dequant(pool_v, sf_v, g_v, slots, torch.bfloat16)
    k = k.view(rows, args.width, KV_HEADS, HEAD_DIM)
    v = v.view(rows, args.width, KV_HEADS, HEAD_DIM)
    groups = QO_HEADS // KV_HEADS
    k = k.repeat_interleave(groups, dim=2)
    v = v.repeat_interleave(groups, dim=2)
    sm_scale = args.sm_scale or 1.0 / (HEAD_DIM ** 0.5)
    scores = torch.einsum("rqd,rktd->rqkt", tokens["q"].float(), k.float()) * sm_scale
    probs = torch.softmax(scores, dim=2)
    out = torch.einsum("rqkt,rktd->rqt", probs, v.float())
    return (out.bfloat16() * torch.sigmoid(tokens["gate"].float()).bfloat16())


def run_flashinfer(tokens: dict, args: argparse.Namespace, device: torch.device,
                   backend: str) -> torch.Tensor:
    from flashinfer.sparse import qsa_sparse_paged_attention

    return qsa_sparse_paged_attention(
        tokens["q"],
        tokens["data_k"],
        tokens["data_v"],
        tokens["logical_indices"],
        tokens["block_table"],
        tokens["token_to_req"],
        False,                       # use_prefill_config: accepted, no effect
        None,                        # out: let the kernel allocate
        args.k_scale,
        args.v_scale,
        output_gate=tokens["gate"],
        k_sf=tokens["sf_k"],
        v_sf=tokens["sf_v"],
        backend=backend,
    )


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def time_calls(fn, reps: int, device: torch.device, flush: torch.Tensor | None) -> list[float]:
    timings: list[float] = []
    for _ in range(reps):
        if flush is not None:
            flush.zero_()
        if device.type == "cuda":
            start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            fn()
            stop.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(stop) * 1000.0)
        else:
            t0 = time.perf_counter()
            fn()
            timings.append((time.perf_counter() - t0) * 1e6)
    return timings


def time_graph(fn, reps: int, device: torch.device) -> float:
    """Median CUDA-graph replay time in microseconds; raises when capture is impossible."""
    if device.type != "cuda":
        raise RuntimeError("graph capture needs CUDA")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
    timings: list[float] = []
    for _ in range(reps):
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        graph.replay()
        stop.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(stop) * 1000.0)
    return statistics.median(timings)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def score(actual: torch.Tensor, truth: torch.Tensor) -> dict:
    a = actual.float()
    t = truth.float()
    if a.shape != t.shape:
        return {"shape_ok": False, "actual_shape": tuple(a.shape), "truth_shape": tuple(t.shape)}
    diff = (a - t).abs()
    denom = t.abs().clamp_min(1e-6)
    return {
        "shape_ok": True,
        "max_abs_err": float(diff.max()),
        "mean_abs_err": float(diff.mean()),
        "max_rel_err": float((diff / denom).max()),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repo:
        sys.path.insert(0, str(Path(args.repo) / "python"))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("WARNING: CUDA unavailable, falling back to the CPU reference path", file=sys.stderr)
    torch.manual_seed(args.seed)

    flush = None
    flush_modes = ("on", "off") if args.l2_flush == "both" else (args.l2_flush,)
    if "on" in flush_modes:
        flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)

    wanted = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in wanted if f not in FAMILIES]
    if unknown:
        raise SystemExit(f"unknown families {unknown}; choose from {FAMILIES}")

    rows_out: list[dict] = []
    rc = 0
    for rows in args.rows:
        tokens = build_case(args, rows, device)
        truth = reference_attention(tokens, args, device)
        base = {"rows": rows, "width": args.width, "context": tokens["context"],
                "q_heads": QO_HEADS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM,
                "page": PAGE, "k_scale": args.k_scale, "v_scale": args.v_scale}
        for flush_mode in flush_modes:
            row_flush = flush if flush_mode == "on" else None
            for family in ["reference"] + wanted:
                rec = dict(base, family=family, l2_flush=flush_mode)
                try:
                    if family == "reference":
                        fn = lambda: reference_attention(tokens, args, device)  # noqa: E731
                    elif family == "ours_bf16":
                        fn = lambda: run_ours_bf16(tokens, args, device)  # noqa: E731
                    elif family == "fi_ref":
                        fn = lambda: run_flashinfer(tokens, args, device, "default")  # noqa: E731
                    else:
                        fn = lambda: run_flashinfer(tokens, args, device, "cake")  # noqa: E731
                    for _ in range(args.warmups):
                        fn()
                    timings = time_calls(fn, args.reps, device, row_flush)
                    rec.update({"status": "ok", "median_us": statistics.median(timings),
                                "min_us": min(timings), "max_us": max(timings),
                                "reps": args.reps})
                    rec.update(score(fn(), truth))
                    if args.capture:
                        try:
                            rec["graph_median_us"] = time_graph(fn, args.reps, device)
                        except Exception as exc:  # capture is a finding, not a crash
                            rec["graph_error"] = f"{type(exc).__name__}: {exc}"
                except Exception as exc:  # noqa: BLE001 - a broken family is a result row
                    rec.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                    rc = rc or (1 if args.fail_fast else 0)
                rows_out.append(rec)
                print(json.dumps(rec, sort_keys=True))
                if args.fail_fast and rec["status"] == "error":
                    break

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows_out))
        print(f"wrote {len(rows_out)} rows to {out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
