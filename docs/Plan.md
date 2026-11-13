# Project plan

Train one **123,489,024-parameter Llama-style model** from random weights on TinyStories. Use the same run from the first pilot through the final pretraining checkpoint. Once it produces coherent story continuations, fine-tune a copy on SST-5 and compare its development accuracy with a majority-class baseline.

Allow **one week at about one hour per day** in Codespaces. Detached Modal jobs can continue between check-ins. The total Modal project budget is **US$200**.

## Course work and adaptations

| Course material | What we do in this project | Adaptation |
| --- | --- | --- |
| Stanford CS336 A1, Basics | Train a byte-level BPE tokenizer, build a decoder-only Transformer, and pretrain it on TinyStories. | Train a fresh vocabulary with Hugging Face Tokenizers. Use PyTorch AdamW and scaled dot-product attention. Keep one model configuration through the pilot and longer run. |
| Stanford CS336 A2, Systems | Measure sustained training throughput and peak allocated GPU memory to plan the next run. | Use a single H100. The one-week scope covers measurement from the training logs and metrics, with distributed training and custom Triton kernels available as follow-up work. |
| CMU 11-711 Fall 2026 HW1, Build a Language Model | Use Llama-style blocks, evaluate on held-out text, sample with temperature and top-p, and adapt the checkpoint to SST-5. | Use raw TinyStories with our own BPE tokenizer and size the run against our Modal budget. The course reference uses a different tokenizer. Use our own validation split and tokenizer for loss comparisons across checkpoints. |

[Stanford CS336](https://cs336.stanford.edu/) · [CMU HW1](https://cmu-anlp.org/assignments/01)

## Schedule

| Day | Work during the session | Evidence to keep |
| --- | --- | --- |
| 1 | Open Codespaces, run CPU tests, check Modal billing headroom, and start dataset preparation. | Commit, test outcome, prep app ID, and Volume manifest once ready. |
| 2 | Check the manifest and launch the 100-step pilot on one H100. | Initial and final validation loss, completed step, throughput, peak GPU memory, and billed usage. |
| 3 | Resume the same run toward step 1,000. Estimate the next segment from observed throughput and spend. | The logged resume step and first usable cost estimate. |
| 4 | Generate samples with fixed prompts and settings. Start the next detached segment if the measurements support it. | Samples with checkpoint step, sampling settings, and planned target. |
| 5 | Review held-out loss and story coherence. Choose the pretraining checkpoint for SST-5. | Loss across checkpoints and examples that show both successes and failures. |
| 6 | Fine-tune on SST-5 and collect development metrics. Repeat with a second seed if time allows. | Source checkpoint step, seed, accuracy by epoch, and majority baseline. |
| 7 | Summarize the results, remaining questions, and actual Modal spend. | Final checkpoint step, processed tokens, loss, sample excerpts, SST-5 result, and billing total. |

The pilot's target is step 100, about 1.64M processed tokens at the default batch size. Extend training only after a committed checkpoint and usable throughput measurements are available. Keep the prompts and decoding settings fixed when comparing checkpoints. For story quality, look for continuation of the premise, consistent characters, and a sequence of events that makes sense.

## Modal budget

| Work | Working allocation (USD) |
| --- | ---: |
| Data preparation and pilot | $20 |
| Main pretraining | $130 |
| SST-5 | $20 |
| Reruns and overhead | $30 |
| **Total** | **$200** |

Treat these as planning allocations. Check remaining workspace usage and spend headroom before the next segment, and reconcile actual project charges from Modal billing. If time or budget runs short, report the last verified checkpoint and the results it supports.

[Runbook](Runbook.md) · [Learning progress](Learning-Progress.md) · [Modal journal](cloud-walker-Modal.md)
