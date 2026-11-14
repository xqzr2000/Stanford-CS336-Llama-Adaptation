# Project plan

Train one **123,489,024-parameter Llama-style model** from random weights on TinyStories. Run a 100-step pilot to measure throughput, choose the schedule length, then keep one run through the final pretraining checkpoint. Once it produces coherent story continuations, fine-tune a copy on SST-5 and compare its development accuracy with a majority-class baseline.

Allow **one week at about one hour per day** in Codespaces. Detached Modal jobs can continue between check-ins. The total Modal project budget is **US$200**.

## Course work and adaptations

| Course material | What we do in this project | Adaptation |
| --- | --- | --- |
| Stanford CS336 A1, Basics | Train a byte-level BPE tokenizer, build a decoder-only Transformer, and pretrain it on TinyStories. | Train a fresh vocabulary with Hugging Face Tokenizers. Use PyTorch AdamW and scaled dot-product attention. Keep one model configuration through the pilot and the main run. |
| Stanford CS336 A2, Systems | Measure sustained training throughput and peak allocated GPU memory to plan the next run. | Use a single H100. The one-week scope covers measurement from the training logs and metrics, with distributed training and custom Triton kernels available as follow-up work. |
| CMU 11-711 Fall 2026 HW1, Build a Language Model | Use Llama-style blocks, evaluate on held-out text, sample with temperature and top-p, and adapt the checkpoint to SST-5. | Use raw TinyStories with our own BPE tokenizer and size the run against our Modal budget. The course reference uses a different tokenizer. Use our own validation split and tokenizer for loss comparisons across checkpoints. |

[Stanford CS336](https://cs336.stanford.edu/) · [CMU HW1](https://cmu-anlp.org/assignments/01)

## Schedule

| Day | Work during the session | Evidence to keep |
| --- | --- | --- |
| 1 | Open Codespaces, run CPU tests, check Modal billing headroom, and start dataset preparation. | Commit, test outcome, prep app ID, and Volume manifest once ready. |
| 2 | Check the manifest and launch the 100-step pilot on one H100. | Initial and final validation loss, completed step, throughput, peak GPU memory, and billed usage. |
| 3 | Fill in the cost estimate from the pilot's throughput and choose the schedule length. Start the main run toward step 1,000. | The chosen `--schedule-steps`, the first main-run checkpoint, and the updated cost estimate. |
| 4 | Generate samples with fixed prompts and settings. Start the next detached segment if the measurements support it. | Samples with checkpoint step, sampling settings, and planned target. |
| 5 | Review held-out loss and story coherence. Choose the pretraining checkpoint for SST-5. | Loss across checkpoints and examples that show both successes and failures. |
| 6 | Fine-tune on SST-5 and collect development metrics. Repeat with a second seed if time allows. | Source checkpoint step, seed, accuracy by epoch, and majority baseline. |
| 7 | Summarize the results, remaining questions, and actual Modal spend. | Final checkpoint step, processed tokens, loss, sample excerpts, SST-5 result, and billing total. |

The pilot's target is step 100, about 1.64M processed tokens at the default batch size. It runs under the run ID `pilot`, and its checkpoint is used for measurement only. Start the main run only after a committed pilot checkpoint and usable throughput measurements are available. Keep the prompts and decoding settings fixed when comparing checkpoints. For story quality, look for continuation of the premise, consistent characters, and a sequence of events that makes sense.

## Cost estimate

The H100 throughput is unmeasured until the pilot. The table shows three assumed rates. Replace them with the pilot's `train_tokens_per_second` from `metrics-000100.json`.

**GPU hours = optimizer steps × 16,384 ÷ tokens per second ÷ 3,600.** The hourly rate used here is about **US$4.20**: the H100 at $3.95 plus 4 CPU cores and 8 GiB of memory at list prices. Check [Modal pricing](https://modal.com/pricing) for current rates.

| `--schedule-steps` | Processed tokens | Passes over the 534M training tokens | At 100k tokens/s | At 200k tokens/s | At 300k tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 15,000 (default) | 246M | 0.46 | 0.7 h · $2.87 | 0.3 h · $1.43 | 0.2 h · $0.96 |
| 32,600 | 534M | 1.00 | 1.5 h · $6.24 | 0.7 h · $3.12 | 0.5 h · $2.08 |
| 65,200 | 1,068M | 2.00 | 3.0 h · $12.47 | 1.5 h · $6.24 | 1.0 h · $4.16 |
| 97,800 | 1,602M | 3.00 | 4.5 h · $18.71 | 2.2 h · $9.35 | 1.5 h · $6.24 |

These figures cover training steps only. Add the following.

- **Checkpoints.** Each one runs validation, writes about 1.5 GB, and commits the Volume. The default interval of 500 steps gives 30 checkpoints over 15,000 steps and 195 over 97,800. Read `validation_and_save_seconds` in the metrics and `checkpoint_seconds` in the log. If checkpoints take a large share of the time, raise `--checkpoint-every`. It can change between segments.
- **Startup.** Each segment pays for container start, loading the data, and loading the checkpoint. Allow a few minutes of H100 time per segment, about $0.30.
- **Other jobs.** Data preparation and sampling run on CPU at about $0.25 per hour. The pilot and each SST-5 run are a few minutes of H100 time each.

**Worst case for a stuck job.** A job that stops making progress fails itself after 15 minutes, about $1. If that guard does not fire, Modal's function timeout ends the job after 6 hours, about $25.

On these assumptions the whole project costs well under the allocations below, which remain the caps. The schedule length is fixed at the first step of the main run, so choose it from this table once the pilot has replaced the assumed rates.

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
