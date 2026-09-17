# 纵向能力成绩单

| 指标 | pretrain:native-60m-baseline-v1 | sft:native-sft-public-60m-001 | dpo:native-dpo-public-60m-001 | grpo:native-grpo-public-60m-001 |
| --- | ---: | ---: | ---: | ---: |
| continuation.nonempty | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| continuation.nonempty.en | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| continuation.nonempty.zh | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| continuation.repeated_trigram | 0.203442 | 0.204167 | 0.158333 | 0.270833 |
| continuation.repeated_trigram.en | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| continuation.repeated_trigram.zh | 0.406884 | 0.408333 | 0.316667 | 0.541667 |
| corpus.bpb.all | 2.035306 | 2.008218 | 2.007856 | 2.009478 |
| corpus.bpb.en | 1.529443 | 1.518601 | 1.518383 | 1.521365 |
| corpus.bpb.zh | 2.589195 | 2.544317 | 2.543797 | 2.543931 |
| corpus.loss.all | 5.694680 | 5.618889 | 5.617876 | 5.622414 |
| corpus.loss.en | 4.585057 | 4.552555 | 4.551903 | 4.560840 |
| corpus.loss.zh | 6.751464 | 6.634444 | 6.633088 | 6.633437 |
| format.success | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| format.success.en | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| format.success.zh | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| instruction.success | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| instruction.success.en | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| instruction.success.zh | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| memory.test.suffix_exact | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| memory.test.suffix_exact.en | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| memory.test.suffix_exact.zh | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| memory.train.suffix_exact | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| memory.train.suffix_exact.en | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| memory.train.suffix_exact.zh | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| multiturn.all_correct | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| multiturn.all_correct.en | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| multiturn.all_correct.zh | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| multiturn.success | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| multiturn.success.en | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| multiturn.success.zh | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| overlap.heldout_fraction | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| overlap.query_fraction | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| preference.accuracy | 0.437500 | 0.437500 | 0.468750 | 0.437500 |
| preference.accuracy.en | 0.500000 | 0.437500 | 0.500000 | 0.437500 |
| preference.accuracy.zh | 0.375000 | 0.437500 | 0.437500 | 0.437500 |
| qa.success | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| qa.success.en | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| qa.success.zh | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| verifiable.reward | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| verifiable.reward.en | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| verifiable.reward.zh | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Delta is current minus first. Negative BPB/repetition delta is better; positive task/reward delta is better. No aggregate capability score.
