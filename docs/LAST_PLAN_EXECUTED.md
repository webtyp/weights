# Plan (executed locally) — `Int8Block32` and `DequantRow`

## Context

`bekko` stores int8 weights with one scale per row. A language model generates text token by
token, and small errors compound, so it needs finer scales: one per block of 32 values, the
layout GGUF `Q8_0` uses (decision D13 of the agent ecosystem master plan). Every model also
repeated its own "row to float32" loop (`bekko/dequant.go`).

## Design gate (api-design)

1. **Prior art.** GGUF `Q8_0` (32 int8 + one fp16 scale per block), ONNX `QuantizeLinear` with
   `block_size`, and PyTorch's `per_channel_group` quantization. All use per-block scales for
   LLMs. We keep float32 scales (the format already stores them) and blocks of 32.
2. **Novice-name test.** `weights.Int8Block32`, `weights.BlockSize`, `Tensor.DequantRow(dst, i)`,
   `Tensor.Cols()`.
3. **Complexity ledger.** Concepts +3 / −0. Ways to do the same thing: model repositories can
   drop their private dequantization loops (−1 each when they migrate).
4. **Where it belongs.** The storage format and how to read it back is this repository's concern.
5. **What it deletes.** Nothing yet. `bekko/dequant.go`'s `dequantRow` can switch to `DequantRow`.
