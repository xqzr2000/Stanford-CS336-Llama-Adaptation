# Project plan

Train one **123,489,024-parameter Llama-style model** from random weights on FineWeb-Edu. Run a 100-step pilot to measure throughput, choose the schedule length, then keep one run through the final pretraining checkpoint. Once its continuations stay on topic, fine-tune a copy on SST-5 and compare its development accuracy with a majority-class baseline.

Allow **one week at about one hour per day** in Codespaces. Detached Modal jobs can continue between check-ins. The total Modal project budget is **US$200**.

## Course work and adaptations

| Course material | What we do in this project | Adaptation |
| --- | --- | --- |
| Stanford CS336 A1, Basics | Train a byte-level BPE tokenizer, build a decoder-only Transformer, and pretrain it on web text. | Use FineWeb-Edu in place of TinyStories and OpenWebText. Train a fresh vocabulary with Hugging Face Tokenizers. Use PyTorch AdamW and scaled dot-product attention. Keep one model configuration through the pilot and the main run. |
| Stanford CS336 A2, Systems | Measure sustained training throughput and peak allocated GPU memory to plan the next run. | Use a single H100. The one-week scope covers measurement from the training logs and metrics, with distributed training and custom Triton kernels available as follow-up work. |
| CMU 11-711 Fall 2026 HW1, Build a Language Model | Use Llama-style blocks, evaluate on held-out text, sample with temperature and top-p, and adapt the checkpoint to SST-5. | Use FineWeb-Edu with our own BPE tokenizer instead of the course's TinyStories data, and size the run against our Modal budget. The course reference uses a different tokenizer. Use our own validation split and tokenizer for loss comparisons across checkpoints. |

[Stanford CS336](https://cs336.stanford.edu/) · [CMU HW1](https://cmu-anlp.org/assignments/01)

## Schedule

| Day | Work during the session | Evidence to keep |
| --- | --- | --- |
| 1 | Open Codespaces, run CPU tests, check Modal billing headroom, and start dataset preparation. | Commit, test outcome, prep app ID, and Volume manifest once ready. |
| 2 | Check the manifest and launch the 100-step pilot on one H100. | Initial and final validation loss, completed step, throughput, peak GPU memory, and billed usage. |
| 3 | Fill in the cost estimate from the pilot's throughput and choose the schedule length. Start the main run toward step 1,000. | The chosen `--schedule-steps`, the first main-run checkpoint, and the updated cost estimate. |
| 4 | Generate samples with fixed prompts and settings. Start the next detached segment if the measurements support it. | Samples with checkpoint step, sampling settings, and planned target. |
| 5 | Review held-out loss and sample quality. Choose the pretraining checkpoint for SST-5. | Loss across checkpoints and examples that show both successes and failures. |
| 6 | Fine-tune on SST-5 and collect development metrics. Repeat with a second seed if time allows. | Source checkpoint step, seed, accuracy by epoch, and majority baseline. |
| 7 | Summarize the results, remaining questions, and actual Modal spend. | Final checkpoint step, processed tokens, loss, sample excerpts, SST-5 result, and billing total. |

The pilot's target is step 100, about 1.64M processed tokens at the default batch size. It runs under the run ID `pilot`, and its checkpoint is used for measurement only. Start the main run only after a committed pilot checkpoint and usable throughput measurements are available. Keep the prompts and decoding settings fixed when comparing checkpoints. For sample quality, look for grammatical sentences that stay on the prompt's topic. Expect fluent but often factually wrong text from a model this size. Validation loss on FineWeb-Edu is higher than on TinyStories and the two are not comparable.

## Cost estimate

The H100 throughput is unmeasured until the pilot. The table shows three assumed rates. Replace them with the pilot's `train_tokens_per_second` from `metrics-000100.json`.

**GPU hours = optimizer steps × 16,384 ÷ tokens per second ÷ 3,600.** The hourly rate used here is about **US$4.33**: the H100 at $3.95 plus 4 CPU cores and 24 GiB of memory at list prices. The trainer holds the training tokens in memory, about 5 GB for the default three shards. Check [Modal pricing](https://modal.com/pricing) for current rates.

The default three training shards hold about 2.5B tokens with our tokenizer. That figure is an estimate until preparation writes the manifest. Hugging Face lists the 10BT sample as 10B GPT-2 tokens over 13¼ shard-sized files, and our 16,384-entry vocabulary needs somewhat more tokens than GPT-2's 50,257 for the same text.

| `--schedule-steps` | Processed tokens | Tokens per parameter | Training shards for one pass | At 100k tokens/s | At 200k tokens/s | At 300k tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50,000 | 0.82B | 7 | 1 | 2.3 h · $9.85 | 1.1 h · $4.93 | 0.8 h · $3.28 |
| 100,000 | 1.64B | 13 | 2 | 4.6 h · $19.71 | 2.3 h · $9.85 | 1.5 h · $6.57 |
| 150,000 (default) | 2.46B | 20 | 3 (default) | 6.8 h · $29.56 | 3.4 h · $14.78 | 2.3 h · $9.85 |
| 300,000 | 4.92B | 40 | 6 | 13.7 h · $59.13 | 6.8 h · $29.56 | 4.6 h · $19.71 |

About 20 tokens per parameter is the compute-optimal ratio reported for Chinchilla. Runs longer than one pass over the prepared shards repeat data. Prepare more shards instead, up to six.

These figures cover training steps only. Add the following.

- **Checkpoints.** Each one runs validation, writes about 1.5 GB, and commits the Volume. The default interval of 2,000 steps gives 75 checkpoints over 150,000 steps. Read `validation_and_save_seconds` in the metrics and `checkpoint_seconds` in the log. If checkpoints take a large share of the time, raise `--checkpoint-every`. It can change between segments.
- **Startup.** Each segment pays for container start, reading about 5 GB of tokens from the Volume, and loading the checkpoint. Allow a few minutes of H100 time per segment, about $0.30. A 150,000-step run needs at least one resume at 330 minutes per segment.
- **Data preparation.** 8 CPU cores and 16 GiB cost about $0.51 per hour. Downloading about 7 GB and tokenizing 2.5B tokens should take under an hour, so well under $1. Sampling runs on CPU at about $0.25 per hour. The pilot and each SST-5 run are a few minutes of H100 time each.
- **Hosting the data.** Hugging Face downloads are free. Data coming into Modal is not billed as egress. The Volume holds about 5 GB of tokens plus a 1.5 GB checkpoint per run, far below the 1 TiB of Volume storage Modal includes each month. Above that, storage is $0.09 per GiB-month. Raw shards are deleted after encoding, so they never reach the Volume. Reading results back with `m volume get` counts as egress, but metrics and samples are a few kilobytes against 1 TiB included.

**Worst case for a stuck job.** A job that stops making progress fails itself after 15 minutes, about $1. If that guard does not fire, Modal's function timeout ends the job after 6 hours, about $25.

At 100k tokens/s the default schedule uses about $30 of the $130 pretraining allocation; at 300k tokens/s, about $10. The allocations below remain the caps. The schedule length is fixed at the first step of the main run, so choose it from this table once the pilot has replaced the assumed rates.

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
