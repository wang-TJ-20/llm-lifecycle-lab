# License and Attribution Notice

## Implementation

The `llm-lifecycle-lab` source code used to train and load this checkpoint is
distributed under the Apache License 2.0:

https://github.com/wang-TJ-20/llm-lifecycle-lab

The Apache-2.0 license for the implementation does not, by itself, assign the
same license to the trained model weights.

## Training Data

The frozen training manifest records the following sources:

1. `SimpleStories/SimpleStories`
   - Revision: `e63b8adc3b1a1bdc7cac5b500d150b71346b0628`
   - Recorded license: MIT
   - Synthetic data: yes
2. `wikimedia/wikipedia`, configuration `20231101.zh`
   - Revision: `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`
   - Recorded license: CC-BY-SA-3.0
   - Synthetic data: no

The combined data manifest records `CC-BY-SA-3.0 AND MIT`.

## Model Weights

The model weights (`model.pt`) and tokenizer artifacts (`tokenizer.json` and
`tokenizer_manifest.json`) are distributed under Creative Commons
Attribution-ShareAlike 4.0 International (`CC-BY-SA-4.0`):

https://creativecommons.org/licenses/by-sa/4.0/legalcode

The implementation remains under Apache-2.0, and the upstream training data
remains under its respective licenses. The model artifact license does not
assert restrictions over model outputs. See `LICENSE_MODEL` for the exact
scope and attribution instructions.

This notice is informational and is not legal advice.
