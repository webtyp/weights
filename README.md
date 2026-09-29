# weights
<img src="docs/img/badges.svg">

The model artifact format of webtyp (`WTYPW1`): named tensors with their shapes, stored so they
can be read without copying, plus the tokenizer vocabulary. `webtyp/weightsc` writes artifacts
from a model's original files, and the model repositories (`bekko`, `decoder`/`qwen`) read them.

## Tensor storage

| `DType` | Storage | Used for |
|---|---|---|
| `Float32` | little-endian float32 | norms, biases, small tensors, test fixtures |
| `Int8` | int8 + **one scale per row** | the embedding model (`bekko`) |
| `Int8Block32` | int8 + **one scale per block of 32 values** of a row (GGUF `Q8_0` layout) | language models, where per-row scales lose too much precision |

`Tensor.DequantRow(dst, i)` returns row `i` as float32 whatever the storage, and `Tensor.Cols()`
gives its length. An `Int8Block32` tensor whose scale count does not match its shape is rejected
by `Open` (`ErrScalesMismatch`).

## Documentation

- [Last executed plan](docs/LAST_PLAN_EXECUTED.md): why `Int8Block32` exists.
- The block layout and why it matters for SIMD: [`webtyp/nn/docs/SIMD.md`](https://github.com/webtyp/nn/blob/main/docs/SIMD.md).
