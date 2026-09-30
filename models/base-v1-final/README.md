---
license: cc-by-sa-4.0
license_link: https://creativecommons.org/licenses/by-sa/4.0/legalcode
language:
- en
- zh
pipeline_tag: text-generation
---

# Native-60M Base v1

Native-60M Base v1 is a 62.9M-parameter bilingual decoder-only language model
trained from random initialization. This repository publishes the final Base
checkpoint and its tokenizer as a reproducible research artifact.

## Model Details

| Field | Value |
| --- | --- |
| Parameters | 62,927,616 |
| Architecture | Dense decoder-only Transformer |
| Layers / hidden size | 8 / 768 |
| Attention heads / KV heads | 12 / 4 |
| Vocabulary | 16,384 byte-level BPE tokens |
| Maximum sequence length | 512 |
| Languages | English and Chinese |
| Training precision | BF16 |
| Final training step | 45,191 |
| Training tokens | 369,480,328 |

The model is a Base language model. It has not been instruction-tuned and
should not be expected to follow chat instructions reliably.

## Files

| File | Purpose |
| --- | --- |
| `model.pt` | PyTorch state dict |
| `config.json` | Native model architecture |
| `tokenizer.json` | Byte-level BPE tokenizer |
| `tokenizer_manifest.json` | Tokenizer identity and chat protocol |
| `provenance.json` | Training, evaluation, source and hash provenance |
| `SHA256SUMS` | Integrity hashes for release files |
| `LICENSE_MODEL` | CC-BY-SA-4.0 terms for the weights and tokenizer |
| `NOTICE.md` | Software and training-data license notices |

## Loading

This is a native `llm-lifecycle-lab` checkpoint. It is not directly compatible
with Hugging Face `AutoModelForCausalLM`.

Install the implementation at the recorded source revision:

```bash
git clone https://github.com/wang-TJ-20/llm-lifecycle-lab.git
cd llm-lifecycle-lab
git checkout 373156d0a0330ceb4aecaa8e4bdeaece8e4f5a9a
python -m pip install -e .
```

Load the downloaded model directory:

```python
from pathlib import Path

from llm_lifecycle_lab.model.native import (
    NativeTransformer,
    load_native_model_config,
)
from llm_lifecycle_lab.tokenizer import NativeTokenizer

model_dir = Path("/path/to/native-60m-base-v1")
tokenizer = NativeTokenizer.from_directory(model_dir)
model = NativeTransformer(
    load_native_model_config(model_dir / "config.json")
)
model.load(model_dir)
model.eval()
```

## Training Data

The frozen `bilingual-60m-v1` corpus contains:

- 100,000 English records from `SimpleStories/SimpleStories`, revision
  `e63b8adc3b1a1bdc7cac5b500d150b71346b0628`.
- 126,000 Chinese records from `wikimedia/wikipedia`, configuration
  `20231101.zh`, revision
  `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`.

The records were mixed round-robin and split deterministically by source ID
with seed 42. SimpleStories is synthetic story data, while the Chinese corpus
is encyclopedic text. These domains are not directly comparable.

## Evaluation

Evaluation used a fixed 1,024-window subset for each split.

| Split | Overall loss | English loss | Chinese loss | Bits/byte |
| --- | ---: | ---: | ---: | ---: |
| Dev, step 0 | 9.859895 | 9.844470 | 9.874070 | 3.664932 |
| Dev, step 45,191 | 2.754549 | 1.953569 | 3.490612 | 1.023868 |
| Test, step 45,191 | 2.731877 | 1.955555 | 3.419416 | 1.022459 |

The final checkpoint passed the preregistered development gate. The best
overall dev loss was 2.706490 at step 33,000; final dev loss was 1.776% above
that value and remained within the fixed 2% rollback limit.

## Limitations

- This is not an instruction-following or chat model.
- Context length is limited to 512 tokens.
- English training data is synthetic and narrow-domain.
- Chinese evaluation loss is materially higher than English evaluation loss.
- The repository uses a custom native architecture and loader.
- The checkpoint should be treated as a research baseline, not a
  production-safety-reviewed model.

## Reproducibility

The source run is `native-60m-base-v1-s42`. Its complete compact evidence bundle
is stored in the source repository under
`results/native-60m-base-v1-s42/`.

The model weight SHA-256 is:

```text
4cdbd642cb878c0e7f5de4b7988751e92de533afa1317256ace20aea078f9398
```

The run started from Git commit
`373156d0a0330ceb4aecaa8e4bdeaece8e4f5a9a`. The runtime recorded a dirty
worktree containing five untracked entries. The recorded source, pipeline,
model, tokenizer and data hashes match the canonical artifacts; this deviation
is retained explicitly in `provenance.json`.

## License and Attribution

The model weights (`model.pt`) and tokenizer artifacts (`tokenizer.json` and
`tokenizer_manifest.json`) are licensed under Creative Commons
Attribution-ShareAlike 4.0 International (`CC-BY-SA-4.0`). Attribute the work as
"Native-60M Base v1" by `wang-TJ-20`, link to the source repository and license,
and indicate whether changes were made. Adapted material must be distributed
under the same license.

The implementation source code remains licensed under Apache-2.0. The training
data remains subject to its upstream MIT and CC-BY-SA-3.0 licenses. The model
artifact license does not assert restrictions over model outputs. See
`LICENSE_MODEL` for the exact scope and legal-code link, and `NOTICE.md` for
upstream attribution.
