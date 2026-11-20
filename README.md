# Stanford CS336 Llama Adaptation

Training small GPT-2-class language models from scratch, following Stanford CS336 and CMU 11-711, on a student budget.

| Track | Model | Status | Start here |
| --- | --- | --- | --- |
| 1. TinyStories on a Colab T4 | 22.7M-parameter Transformer | **Completed.** 327.68M training tokens, **1.4099** full-split validation loss | [Results](CS336_Colab/README.md) · [Trained notebook](CS336_Colab/CS336_TinyStories_Trained.ipynb) · [Model release](https://github.com/xqzr2000/Stanford-CS336-Llama-Adaptation/releases/tag/tinystories-v1) |
| 2. FineWeb-Edu, tested on Colab, trained on Modal | 124.7M-parameter Llama | **Planned.** Smoke test and LR sweep on a Colab A100 or L4, then the full run on a Modal H100 | [Section 3](#3-track-2-a-124m-llama-on-fineweb-edu) |

## 1. Where this project sits in the GPT-2 family

The GPT-2 paper ([Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)) describes four models with the same architecture at different depths and widths. OpenAI released the smallest one first, staged the middle two over the following months, and released the full 1.5B model in November 2019 after withholding it over misuse concerns.

| Model | Parameters (paper) | Parameters (released) | Layers | Width | Heads |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-2 Small | 117M | 124M | 12 | 768 | 12 |
| GPT-2 Medium | 345M | 355M | 24 | 1,024 | 16 |
| GPT-2 Large | 762M | 774M | 36 | 1,280 | 20 |
| GPT-2 XL | 1,542M | 1,558M | 48 | 1,600 | 25 |

Reproductions use the released counts, so "GPT-2 124M" means GPT-2 Small.

This project trains two models:

- **23M (Track 1).** Smaller than any GPT-2. It validates the whole pipeline (tokenizer, data loading, training loop, checkpoints, evaluation) where a run takes hours on a free-tier-class GPU and every bug is cheap to find.
- **124M (Track 2).** The same 12 layers and width 768 as GPT-2 Small, with Llama-style blocks. This puts the project on the first rung of the GPT-2 family, and 124M on about 10B FineWeb-Edu tokens is a well-known reference setting (Karpathy's nanoGPT and llm.c), so we can tell whether our training is healthy.

### The 1.5B model is where the real fun begins

As Yann LeCun wrote on [X](https://x.com/ylecun/status/2099248236074545576):

> "Right. Dario was already claiming that GPT2 was too dangerous to open source back in 2019. I made fun of them then. Everyone should make fun of them now."

Sadly, GPT-2 XL is far *too dangerous* for my wallet. After overspending on H100 time for my CMU 11-768 Assignment 3 project, my budget was devastated enough that I had to ~~skip GTA VI~~.

GPT-2 XL is the size that would teach something genuinely new:

- **Time forces multiple GPUs.** The model and its AdamW state (about 25–28 GB) still fit on one 80 GB GPU, but one H100 would need 8–11 days. A practical run uses an 8-GPU node, which means distributed data parallelism and sharded optimizer state.
- **Data becomes its own project.** About 30B tokens at the Chinchilla ratio of 20 tokens per parameter, and often 100B in practice. That's 60–200 GB of tokens on disk and a sample far larger than FineWeb-Edu `sample/10BT`.
- **Operations matter.** A multi-day run needs automatic resume, monitoring, loss-spike recovery, and about 25 GB per checkpoint.

#### What it would cost on Modal

The best measured reference is Karpathy's [llm.c reproduction of GPT-2 1.5B](https://github.com/karpathy/llm.c/discussions/677): 32B tokens on one 8×H100 node in 24 hours, for USD 672 at the rental price he used. That works out to about 40% model FLOPs utilization (MFU), which is very good. Our PyTorch code will likely reach 30–40%.

Training compute is about 6 × parameters × tokens. Modal lists the H100 at USD 3.95 per GPU-hour and the A100 80 GB at USD 2.50 ([pricing](https://modal.com/pricing)).

| Plan | Tokens | Compute | H100-hours | Wall clock on 8×H100 | GPU cost on Modal |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chinchilla-optimal | 30B | 2.8 × 10²⁰ FLOPs | 200–260 | 25–33 h | **USD 780–1,040** |
| Common practice | 100B | 9.3 × 10²⁰ FLOPs | 660–880 | 3.5–4.5 days | **USD 2,600–3,500** |

The same 30B-token run on A100 80 GBs would take about 620 A100-hours and cost about USD 1,560. Add roughly 20–40% for a short LR check, a pilot, failed starts, CPU and storage. **A single 30B-token attempt comes to roughly USD 1,000–1,400**, more than 20 times this project's budget, plus the multi-GPU code we haven't written yet.

For comparison, OpenAI's original training is estimated at about USD 43,000 in 2019 ([nanochat](https://github.com/karpathy/nanochat)). Newer recipes such as nanochat and [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) now reach GPT-2 XL's benchmark scores for under USD 250, and nanochat for about USD 50–75, by using different optimizers and architecture changes. That's a different exercise from training GPT-2 XL itself, and it also needs an 8-GPU node.

**We keep 1.5B as a future plan for when it's affordable.** Steps then: extend the data pipeline to `sample/100BT`, add DDP with sharded optimizer state, test on an 8-GPU Modal container, and run a GPT-2 Medium-sized rehearsal before the full run.

### Why we skip the 345M model

GPT-2 Medium is affordable: about 7B tokens at the Chinchilla ratio, roughly 12 H100-hours or about USD 50 on Modal. But it teaches nothing new. Same architecture, same tokenizer, same optimizer, same single GPU, just more layers and a longer run for a somewhat lower loss. For a project about learning how these models are built and trained, USD 50 buys more as a careful 124M run. 345M only becomes useful again as a cheap rehearsal before a 1.5B run.

## 2. Track 1: a 23M model on TinyStories (done)

This track follows Stanford CS336 Assignment 1. We trained a 10,000-entry byte-level BPE tokenizer and a 22,696,448-parameter Transformer (4 layers, width 512, RMSNorm, RoPE, SwiGLU) from random weights on a Colab T4.

| Run | Peak LR | Tokens | Full-split validation loss |
| --- | ---: | ---: | ---: |
| Best of a 5-rate sweep | `3e-3` | 40.96M | 1.6544 |
| Full run | `3e-3` | 327.68M | **1.4099** (perplexity 4.10) |

The T4 has no bf16, so the run used FP16 autocast with a gradient scaler. That worked at this size. Section 4 explains why we don't repeat it for 124M.

Full results, curves, sample stories and the Colab demo are in [`CS336_Colab/`](CS336_Colab/README.md). [Open the trained notebook in Colab](https://colab.research.google.com/github/xqzr2000/Stanford-CS336-Llama-Adaptation/blob/main/CS336_Colab/CS336_TinyStories_Trained.ipynb) to generate a story from the released weights on a CPU runtime.

## 3. Track 2: a 124M Llama on FineWeb-Edu

### Model

The architecture comes from CMU 11-711's [Build Your Own Llama](https://cmu-l3.github.io/anlp-fall2025/assignments/assignment1) assignment, which implements Llama 2 blocks (RMSNorm, RoPE, SwiGLU, grouped-query attention). We scale it to GPT-2 Small's depth and width.

| Setting | Value |
| --- | --- |
| Layers, width | 12, 768 |
| Attention | 12 query heads, 4 key/value heads (grouped-query), head size 64, RoPE (θ = 10,000) |
| Feed-forward | SwiGLU, width 2,048 (Llama's 8/3 × width) |
| Normalization | Pre-norm RMSNorm, no biases |
| Vocabulary | 32,000-entry BPE trained on FineWeb-Edu, the same size as Llama 2's |
| Embeddings | Separate input and output matrices |
| Context | 1,024 tokens |
| Parameters | **124,668,672** (75.5M in the blocks, 49.2M in the two embedding matrices) |

Separate embeddings with a 32k vocabulary land at almost exactly GPT-2 Small's 124M. Because the tokenizers differ, our validation loss isn't directly comparable to GPT-2's. We compare against GPT-2 on bits per byte or a benchmark such as HellaSwag instead.

### Data

[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) `sample/10BT`, tokenized once and stored as a flat `uint16` token stream with a manifest and the tokenizer.

- **Already built in Colab:** a 1B-token training cache and 10M validation tokens. That's plenty for the smoke test and the sweep. The context length can change without re-tokenizing.
- **For the main run:** the rest of `sample/10BT` is tokenized on Modal CPUs with the **same `tokenizer.json`**, checked by its SHA-256. About 10B tokens is roughly one pass over the sample.

### Training setup

| Setting | Value |
| --- | --- |
| Precision | bf16 autocast, FP32 weights and optimizer state, TF32 matmuls |
| Effective batch | 512 sequences × 1,024 tokens = **524,288 tokens per step**, through gradient accumulation |
| Optimizer | Fused AdamW, β = (0.9, 0.95), weight decay 0.1, gradient clipping 1.0 |
| Speed | `torch.compile`, flash attention through `scaled_dot_product_attention` |
| Schedule | WSD: 250 warmup steps, constant LR, then linear decay over the last ~20% of tokens |
| Token budget | 2.5B guaranteed (step 4,768), 10B target (step 19,073), within USD 50 |

**Why WSD (warmup–stable–decay) instead of cosine.** A cosine schedule has to know its end point in advance. With WSD the learning rate stays constant until we choose to decay, so:

1. At 2.5B tokens, the Chinchilla-optimal point for 124M, we save the pre-decay checkpoint and branch a short decay (0.25–0.5B tokens). That gives a finished 2.5B model.
2. The main run continues from the **pre-decay** checkpoint, not the decayed branch.
3. We choose when to start the final decay from the measured cost, so the run ends on budget whatever the real throughput turns out to be.

## 4. Smoke test, probe and LR sweep on Colab (A100 or L4, bf16)

### Why not the T4

The T4 has no bf16, so it trains in FP16 with a gradient scaler. FP16 has a much smaller range than bf16. At high learning rates, gradients can overflow even though the same run would be stable in bf16, and the scaler then skips those steps. In a short LR sweep, that pushes the choice too low and adds noise to the warmup. The T4 is also roughly 5–7 times slower than an A100. It served Track 1 well, but a sweep for a bf16 run has to run in bf16.

bf16 keeps FP32's exponent range with fewer mantissa bits. Large values don't overflow and no loss scaling is needed. The L4, A100 and H100 all support it, so the sweep and the final run share the same numerics, and the chosen LR carries over.

| GPU | bf16 | Memory | Bandwidth | Peak dense bf16 | Role |
| --- | --- | --- | --- | --- | --- |
| T4 | ❌ (FP16 only) | 16 GB | 320 GB/s | — (65 TFLOPS FP16) | Track 1 only |
| L4 | ✅ | 24 GB | 300 GB/s | ~121 TFLOPS | Colab fallback when no A100 is available |
| A100 40 GB | ✅ | 40 GB | 1.6 TB/s | 312 TFLOPS | Preferred Colab GPU |
| H100 80 GB | ✅ | 80 GB | 3.35 TB/s | 989 TFLOPS | Final run on Modal |

The L4's problem is speed, not precision. Its memory bandwidth is a fifth of the A100's, so expect about a quarter of the A100's throughput. Colab doesn't guarantee an A100, so the notebook plans for both and sizes the sweep to whichever GPU it gets.

### What the Colab notebook does

1. **Load the cache.** Copy the token cache from Drive to the VM's local disk, check the tokenizer hash, manifest and file sizes, and memory-map the streams.
2. **Smoke test.** Confirm bf16 is supported, overfit a single batch, run a short training segment and check that loss falls without NaNs, and save, reload and resume a checkpoint with identical results.
3. **Probe.** Try micro-batches (for example 32, 16 and 8 sequences of 1,024 tokens) with full optimizer steps. Record tokens per second and peak memory, and pick the fastest one with at least 1 GiB of headroom. Gradient accumulation keeps the effective batch at 524,288 tokens.
4. **LR sweep.** Run the exact final setup (bf16, `torch.compile`, context 1,024, the same batch and warmup) at five peak rates on a log scale: `3e-4, 6e-4, 1e-3, 2e-3, 3e-3`. The LR stays constant after warmup, as in WSD's stable phase. Compare validation loss at matched tokens and watch for loss spikes. Short runs favor slightly higher rates, so if two are close, pick the lower one.
5. **Save** the results, curves and probe numbers to Drive, with hourly resumable checkpoints in case Colab disconnects.

| | A100 40 GB | L4 |
| --- | --- | --- |
| Micro-batch (to be confirmed by the probe) | ~32 × 1,024, 16 accumulation steps | ~8–16 × 1,024, 32–64 accumulation steps |
| Expected speed | ~150k tokens/s | ~40k tokens/s |
| Sweep length | 5 rates × 600 steps (315M tokens each) | 5 rates × 200 steps (105M tokens each) |
| Sweep time (estimate) | ~3 h | ~3.5–4 h |
| Follow-up | None | Recheck the top two rates for 300 steps each on the Modal H100 (about USD 1–2) |

All speeds and times above are estimates. The probe replaces them with measured numbers.

## 5. Final training on a Modal H100

We first planned the full run on a Modal A100 to save money. Pricing it out showed the H100 doesn't cost more for the same tokens:

- Modal charges USD 3.95/h for an H100 and USD 2.10/h for an A100 40 GB, a ratio of 1.9.
- For a model this size, an H100 runs roughly 2–2.5 times faster than an A100. (llm.c's GPT-2 124M run took 90 minutes on 8×A100 and 45 minutes on 8×H100.)
- So the cost per token is about the same, and the run finishes in about half the wall-clock time, with fewer segments to babysit.
- The sweep's bf16 numerics match, so its LR carries over. A short pilot on the H100 measures the throughput that sets the decay point.

### Run plan

1. **Tokenize** the rest of `sample/10BT` on Modal CPUs with the Colab tokenizer, into a Modal Volume.
2. **Pilot:** about 200 H100 steps to measure tokens/s, memory and the cost per billion tokens. On the L4 path, also recheck the top two LRs.
3. **Main run to 2.5B tokens** (step 4,768) in resumable segments. Save the pre-decay checkpoint and run the short decay branch.
4. **Continue the stable phase** from the pre-decay checkpoint, then start the final decay early enough to finish within budget.
5. **Export** FP16 weights, the tokenizer and metrics for the Colab notebook.

### Budget (USD 50 cap)

Set a USD 50 workspace budget in Modal before the first job. The estimates assume 300–350k tokens/s on the H100 (about 22–25% MFU) plus about USD 0.3–0.6/h for the container's CPU and memory.

| Item | Estimate |
| --- | ---: |
| Tokenize ~10B tokens on Modal CPUs | USD 3–5 |
| H100 pilot | ~USD 1 |
| Main run to 2.5B + decay branch | USD 9–11 |
| Continue to 10B with the final decay | USD 26–32 |
| Volume storage (~25 GB) | ~USD 2/month |
| **Total** | **~USD 41–50** |

**Decision rule:** if the pilot measures under about 300k tokens/s, start the final decay earlier (around 7–8B tokens). WSD makes that a one-line change instead of a restart. The 2.5B model is guaranteed in every case.

> **Note:** the code in `llama_lab/` and `modal_app.py` still implements the earlier H100 plan (16,384-entry vocabulary, context 256, 16,384 tokens per step, cosine schedule). It will be updated to this configuration before any paid run. Until then, the [Runbook](docs/Runbook.md) commands describe the old setup.

## 6. Later: SST-5 fine-tuning and post-training

After pretraining, the plan is to fine-tune a copy of the 124M model on SST-5 sentiment classification, as in CMU 11-711. It uses a five-way classification head and compares against the majority-class baseline, on Colab. Post-training (SFT, then DPO) could follow later. At 124M it costs a few dollars on a small Modal GPU, and the point is learning the pipeline, not a useful assistant.

## Project docs

- [Plan](docs/Plan.md) · [Runbook](docs/Runbook.md): written for the earlier H100 plan, to be updated
- [Learning progress](docs/Learning-Progress.md): decisions, checks, experiments and fixes
- [Cloud walker — Modal](docs/cloud-walker-Modal.md): cloud runs, billing, and lessons from 11-768
- [Colab companion notebook](notebooks/CS336_Llama_Project_Walkthrough.ipynb): walks through the math and loads the final checkpoint

## Course adaptations

- **Stanford CS336, Assignment 1.** BPE tokenizer, decoder-only Transformer, AdamW, training loop with checkpoints. Track 1 follows it on TinyStories. Track 2 uses FineWeb-Edu in place of OpenWebText. [CS336](https://cs336.stanford.edu/) · [Assignment 1](https://github.com/stanford-cs336/assignment1-basics)
- **CMU 11-711, Build Your Own Llama.** Llama 2 blocks, sampling with temperature and top-p, SST-5 fine-tuning. We pretrain our own checkpoint on FineWeb-Edu instead of loading the course's TinyStories reference model, and use our own tokenizer throughout.
