
# Introduction

This recipe includes some different ASR models trained with WenetSpeech.

[./RESULTS.md](./RESULTS.md) contains the latest results.

# Transducers

There are various folders containing the name `transducer` in this folder.
The following table lists the differences among them.

|                                       | Encoder             | Decoder            | Comment                     |
|---------------------------------------|---------------------|--------------------|-----------------------------|
| `pruned_transducer_stateless2`        | Conformer(modified) | Embedding + Conv1d | Using k2 pruned RNN-T loss  |                      |
| `pruned_transducer_stateless5`        | Conformer(modified) | Embedding + Conv1d | Using k2 pruned RNN-T loss  |                      |

The decoder in `transducer_stateless` is modified from the paper
[Rnn-Transducer with Stateless Prediction Network](https://ieeexplore.ieee.org/document/9054419/).
We place an additional Conv1d layer right after the input embedding layer.
