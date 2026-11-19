# Draft token maps

`hyudryu_top49152.{json,pt}`: static 49,152-token draft vocabulary for the
Qwen3.8-Flash-Next MTP draft, from
https://github.com/hyudryu/Qwen3.8-Flash-Next-FP4-1xRTXPro6000 (V3, MIT; see
`hyudryu_LICENSE`). The `.pt` file is the same id list saved with `torch.save`
for `--speculative-token-map`.

`examples/run-rc2-production.sh` passes it by default (`TOKEN_MAP`). On top,
speedy keeps a self-learning ring of recently used prompt and accepted-output
tokens (`SGLANG_SPEC_DYNAMIC_VOCAB_SLOTS=8192`, QFN-0099/QFN-0148).
