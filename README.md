# Stanford CS336 Llama Adaptation

Training small GPT-2-class language models from scratch, following Stanford CS336 and CMU 11-711, on a student budget.

| Track | Model | Status | Start here |
| --- | --- | --- | --- |
| 1. TinyStories on a Colab T4 | 22.7M-parameter Transformer | **Completed.** 327.68M training tokens, **1.4099** full-split validation loss | [Results](CS336_Colab/README.md) · [Trained notebook](CS336_Colab/TinyStories/CS336_TinyStories_Trained.ipynb) · [Model release](https://github.com/xqzr2000/Stanford-CS336-Llama-Adaptation/releases/tag/tinystories-v1) |
| 2. FineWeb-Edu on a Colab L4 (bf16) | 124.7M-parameter Llama | **Stage 1 done.** 3.0B training tokens, **3.126** full-split validation loss. Stage 2 (10B tokens) in progress | [Section 3](#3-track-2-a-124m-llama-on-fineweb-edu) · [Results](CS336_Colab/README.md#fineweb-edu) · [Stage 1 notebook](CS336_Colab/FineWebEdu/CS336_FineWebEdu_Llama124M_Trained_Stage_1.ipynb) |
| 3. GPT-2 XL class on a Modal 8×H100 node | ~1–1.4B parameters, modern recipe | **Planned.** Reach GPT-2 XL's CORE score (0.2565), then SFT into a chat model, for about USD 180–290 in capped phases | [Section 7](#7-track-3-a-gpt-2-xl-class-model-on-modal-planned) · [Plan](docs/Plan.md) |

## 1. Where this project sits in the GPT-2 family

The GPT-2 paper ([Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)) describes four models with the same architecture at different depths and widths. OpenAI released the smallest one first, staged the middle two over the following months, and released the full 1.5B model in November 2019 after withholding it over misuse concerns.

| Model | Parameters (paper) | Parameters (released) | Layers | Width | Heads |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-2 Small | 117M | 124M | 12 | 768 | 12 |
| GPT-2 Medium | 345M | 355M | 24 | 1,024 | 16 |
| GPT-2 Large | 762M | 774M | 36 | 1,280 | 20 |
| GPT-2 XL | 1,542M | 1,558M | 48 | 1,600 | 25 |

Reproductions use the released counts, so "GPT-2 124M" means GPT-2 Small.

This project trains three models:

- **23M (Track 1).** Smaller than any GPT-2. It validates the whole pipeline (tokenizer, data loading, training loop, checkpoints, evaluation) where a run takes hours on a free-tier-class GPU and every bug is cheap to find.
- **124M (Track 2).** The same 12 layers and width 768 as GPT-2 Small, with Llama-style blocks. This puts the project on the first rung of the GPT-2 family, and 124M on about 10B FineWeb-Edu tokens is a well-known reference setting (Karpathy's nanoGPT and llm.c), so we can tell whether our training is healthy.
- **GPT-2 XL class (Track 3, planned).** About 1–1.4B parameters trained to GPT-2 XL's benchmark score with a modern recipe on one Modal 8×H100 node. It's the first track that needs multiple GPUs.

### The 1.5B model is where the real fun begins

GPT-2 XL, the 1.5B model, is a landmark. Its paper ([Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)) made the case that a language model trained only to predict the next word of web text starts doing tasks it was never trained for: answering questions, translating, summarizing. It did them badly, but it did them, and it did them better the bigger the model got. GPT-3, ChatGPT and today's assistants all grew from that idea.

It's also the model that brought the argument about releasing AI models into the open. As Yann LeCun wrote on [X](https://x.com/ylecun/status/2099248236074545576) in 2026:

> "Right. Dario was already claiming that GPT2 was too dangerous to open source back in 2019. I made fun of them then. Everyone should make fun of them now."

That's LeCun's view. What actually happened: in February 2019 OpenAI [released only its smallest GPT-2](https://openai.com/index/better-language-models/), citing concerns about the model being used to generate deceptive, biased or abusive language at scale, and called the decision an experiment it wasn't sure was right. Dario Amodei, now Anthropic's CEO, was then OpenAI's VP of Research and a co-author of the paper. The release was staged over the year, and the [full 1.5B model came out in November 2019](https://openai.com/index/gpt-2-1-5b-release/), with OpenAI reporting no strong evidence of misuse so far. Whether that caution was wise, and how it applies to today's far more capable models, is still debated.

**Why it's the goal here.** Tracks 1 and 2 are my learning path through Stanford CS336 and CMU 11-711: tokenizer, Transformer, training loop, learning-rate sweeps, schedules, checkpoints, at 23M and then 124M parameters. Once the basics work, the natural next step is the model that started it all, and the first glimpse of general ability from nothing but next-word prediction.

To be honest about what that glimpse is: a 1.5B-class model is the start of that road, not the destination. It writes fluent paragraphs and knows some facts, but it doesn't reason. "Emergent abilities", skills that jump from chance to real performance once a model is big enough ([Wei et al., 2022](https://arxiv.org/abs/2206.07682)), mostly show up in far larger models, and some researchers argue the jumps depend on how they're measured ([Schaeffer et al., 2023](https://arxiv.org/abs/2304.15004)). What this repo can show is the GPT-2 effect itself: the same kind of model doing tasks it was never trained for, and how that improves along the ladder 23M → 124M → 1.5B.

Sadly, GPT-2 XL is far *too dangerous* for my wallet. After overspending on H100 time for my CMU 11-768 Assignment 3 project, my budget was devastated enough that I had to ~~skip GTA VI~~.

#### Same size, very different training

What makes a model capable isn't its size alone but how much it's trained: parameters × tokens, plus the quality of the data and the recipe. Today's capable 1–2B models are about GPT-2 XL's size, but trained on roughly a thousand times more tokens than a GPT-2-level run:

| Model | Year | Parameters | Training tokens | Tokens per parameter | Training cost |
| --- | --- | ---: | ---: | ---: | ---: |
| GPT-2 XL (OpenAI) | 2019 | 1.5B | not published | — | ~USD 43,000 ([estimate](https://github.com/karpathy/nanochat)) |
| GPT-2 XL reproduction, [llm.c](https://github.com/karpathy/llm.c/discussions/677) | 2024 | 1.6B | 32B | ~20 | USD 672 |
| [nanochat](https://github.com/karpathy/nanochat), GPT-2 XL-level | 2026 | ~1–1.4B | ~7–15B | ~8–10.5 | ~USD 48 (2 h on 8×H100 at USD 3/GPU-hour) |
| **This project, Track 3 (planned)** | 2026–27 | ~1–1.4B | ~7–15B | ~8–10.5 | **USD 180–290** (all phases, on Modal) |
| Llama 3.2 1B (Meta) | 2024 | 1.2B | 9T | ~7,300 | — |
| Qwen2.5-1.5B (Alibaba) | 2024 | 1.5B | 18T | ~12,000 | — |
| SmolLM2-1.7B (Hugging Face) | 2025 | 1.7B | 11T | ~6,500 | ~USD 250,000 |

Token counts for Llama 3.2, Qwen2.5 and SmolLM2, and SmolLM2's cost, are from the [SmolLM2 paper](https://arxiv.org/abs/2502.02737). Track 3 sits at the GPT-2 end of this table on purpose: it's what one learner can afford. The gap to the bottom rows is mostly training data and compute, not the architecture.

GPT-2 XL is the size that would teach something genuinely new:

- **Time forces multiple GPUs.** The model and its AdamW state (about 25–28 GB) still fit on one 80 GB GPU, but one H100 would need 8–11 days. A practical run uses an 8-GPU node, which means distributed data parallelism and sharded optimizer state.
- **Data becomes its own project.** About 30B tokens at the Chinchilla ratio of 20 tokens per parameter, and often 100B in practice. That's 60–200 GB of tokens on disk and a sample far larger than FineWeb-Edu `sample/10BT`.
- **Operations matter.** A multi-day run needs automatic resume, monitoring, loss-spike recovery, and about 25 GB per checkpoint.

#### What a classic reproduction would cost on Modal

The best measured reference is Karpathy's [llm.c reproduction of GPT-2 1.5B](https://github.com/karpathy/llm.c/discussions/677): 32B tokens on one 8×H100 node in 24 hours, for USD 672 at the rental price he used. That works out to about 40% model FLOPs utilization (MFU), which is very good. Our PyTorch code will likely reach 30–40%.

Training compute is about 6 × parameters × tokens. Modal lists the H100 at USD 3.95 per GPU-hour and the A100 80 GB at USD 2.50 ([pricing](https://modal.com/pricing)).

| Plan | Tokens | Compute | H100-hours | Wall clock on 8×H100 | GPU cost on Modal |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chinchilla-optimal | 30B | 2.8 × 10²⁰ FLOPs | 200–260 | 25–33 h | **USD 780–1,040** |
| Common practice | 100B | 9.3 × 10²⁰ FLOPs | 660–880 | 3.5–4.5 days | **USD 2,600–3,500** |

The same 30B-token run on A100 80 GBs would take about 620 A100-hours and cost about USD 1,560. Add roughly 20–40% for a short LR check, a pilot, failed starts, CPU and storage. **A single 30B-token attempt comes to roughly USD 1,000–1,400**, more than 20 times this project's budget, plus the multi-GPU code we haven't written yet.

For comparison, OpenAI's original training is estimated at about USD 43,000 in 2019 ([nanochat](https://github.com/karpathy/nanochat)).

#### The modern route: Track 3

Newer recipes such as [nanochat](https://github.com/karpathy/nanochat) and [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) reach GPT-2 XL's benchmark scores with far less compute. They use the Muon optimizer, modern block details and fewer, better-used tokens. nanochat gets there in 1.65–3 hours on one 8×H100 node, about USD 40–100 depending on the hourly rate.

**That's the plan for Track 3**: GPT-2 XL's capability rather than its exact recipe, in our own code, on Modal. We still get the multi-GPU training, the larger data pipeline and the long-run operations listed above, plus a chat model at the end, for roughly a fifth of the classic cost. [Section 7](#7-track-3-a-gpt-2-xl-class-model-on-modal-planned) summarizes it, and [docs/Plan.md](docs/Plan.md) has the details.

### Why we skip the 345M model

GPT-2 Medium is affordable: about 7B tokens at the Chinchilla ratio, roughly 12 H100-hours or about USD 50 on Modal. But it teaches nothing new. Same architecture, same tokenizer, same optimizer, same single GPU, just more layers and a longer run for a somewhat lower loss. For a project about learning how these models are built and trained, USD 50 buys more as a careful 124M run. The rehearsal before a big run is still worth doing, but in Track 3's recipe it's a small-depth run that costs a few dollars.

## 2. Track 1: a 23M model on TinyStories (done)

This track follows Stanford CS336 Assignment 1. We trained a 10,000-entry byte-level BPE tokenizer and a 22,696,448-parameter Transformer (4 layers, width 512, RMSNorm, RoPE, SwiGLU) from random weights on a Colab T4.

| Run | Peak LR | Tokens | Full-split validation loss |
| --- | ---: | ---: | ---: |
| Best of a 5-rate sweep | `3e-3` | 40.96M | 1.6544 |
| Full run | `3e-3` | 327.68M | **1.4099** (perplexity 4.10) |

The T4 has no bf16, so the run used FP16 autocast with a gradient scaler. That worked at this size. Section 4 explains why we don't repeat it for 124M.

Full results, curves, sample stories and the Colab demo are in [`CS336_Colab/`](CS336_Colab/README.md#tinystories). [Open the trained notebook in Colab](https://colab.research.google.com/github/xqzr2000/Stanford-CS336-Llama-Adaptation/blob/main/CS336_Colab/TinyStories/CS336_TinyStories_Trained.ipynb) to generate a story from the released weights on a CPU runtime.

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

- **Built once in Colab, on a CPU runtime:** the first two `sample/10BT` files (about 1.5B training tokens) and 10M validation tokens from the last file, which training never reads. That's plenty for the smoke test and the sweep. The context length can change without re-tokenizing.
- **For the full run:** the rest of `sample/10BT` was tokenized in Colab with the **same `tokenizer.json`**, checked by its SHA-256, in 1 hour 40 minutes. That gives 9.82B training tokens in 13 files, so 10B tokens is about one pass over the sample.

### Training setup

| Setting | Value |
| --- | --- |
| Precision | bf16 autocast, FP32 weights and optimizer state, TF32 matmuls |
| Effective batch | 512 sequences × 1,024 tokens = **524,288 tokens per step**, through gradient accumulation |
| Optimizer | Fused AdamW, β = (0.9, 0.95), weight decay 0.1, gradient clipping 1.0 |
| Speed | `torch.compile`, flash attention through `scaled_dot_product_attention` |
| Schedule | WSD: 250 warmup steps, constant LR, then linear decay over the last ~20% of tokens |
| Token budget | Stage 1: 2.5B, plus a 0.5B decay branch (done). Stage 2: 10B (step 19,073) |

**Why WSD (warmup–stable–decay) instead of cosine.** A cosine schedule has to know its end point in advance. With WSD the learning rate stays constant until we choose to decay, so:

1. At 2.5B tokens, the Chinchilla-optimal point for 124M, we save the pre-decay checkpoint and branch a short decay (0.5B tokens). That gives a finished 2.5B model: Stage 1.
2. The main run continues from the **pre-decay** checkpoint, not the decayed branch.
3. We choose when to start the final decay from the measured speed, so the run can be shortened at any point before the decay starts.

## 4. Smoke test, probe and LR sweep on Colab (A100 or L4, bf16)

### Why not the T4

The T4 has no bf16, so it trains in FP16 with a gradient scaler. FP16 has a much smaller range than bf16. At high learning rates, gradients can overflow even though the same run would be stable in bf16, and the scaler then skips those steps. In a short LR sweep, that pushes the choice too low and adds noise to the warmup. The T4 is also roughly 5–7 times slower than an A100. It served Track 1 well, but a sweep for a bf16 run has to run in bf16.

bf16 keeps FP32's exponent range with fewer mantissa bits. Large values don't overflow and no loss scaling is needed. The L4, A100 and H100 all support it, so the sweep and the final run share the same numerics, and the chosen LR carries over.

| GPU | bf16 | Memory | Bandwidth | Peak dense bf16 | Role |
| --- | --- | --- | --- | --- | --- |
| T4 | ❌ (FP16 only) | 16 GB | 320 GB/s | — (65 TFLOPS FP16) | Track 1 only |
| L4 | ✅ | 24 GB | 300 GB/s | ~121 TFLOPS | Colab fallback when no A100 is available |
| A100 40 GB | ✅ | 40 GB | 1.6 TB/s | 312 TFLOPS | Preferred Colab GPU |
| H100 80 GB | ✅ | 80 GB | 3.35 TB/s | 989 TFLOPS | Not used for 124M (see Section 5) |

The L4's problem is speed, not precision. Its memory bandwidth is a fifth of the A100's, so expect about a quarter of the A100's throughput. Colab doesn't guarantee an A100, so the notebook plans for both and sizes the sweep to whichever GPU it gets.

### What the Colab notebook does

[`CS336_Colab/FineWebEdu/CS336_FineWebEdu_Llama124M_LR_Sweep.ipynb`](CS336_Colab/FineWebEdu/CS336_FineWebEdu_Llama124M_LR_Sweep.ipynb)

1. **Build the cache once, then load and check it.** On a CPU runtime, the first run downloads FineWeb-Edu shards, trains the 32k BPE tokenizer, tokenizes about 1.5B training and 10M validation tokens, and saves them to Drive. Later sessions copy the cache from Drive to the VM's local disk, check the tokenizer hash, manifest, file sizes and fingerprints, and memory-map the streams.
2. **Smoke test.** Confirm bf16 is supported, overfit a single batch, run a short training segment and check that loss falls without NaNs, and save, reload and resume a checkpoint with identical results.
3. **Probe.** Try micro-batches (for example 32, 16 and 8 sequences of 1,024 tokens) with full optimizer steps. Record tokens per second and peak memory, and pick the fastest one with at least 1 GiB of headroom. Gradient accumulation keeps the effective batch at 524,288 tokens.
4. **LR sweep.** Run the exact final setup (bf16, `torch.compile`, context 1,024, the same batch and warmup) at five peak rates on a log scale on an A100: `3e-4, 6e-4, 1e-3, 2e-3, 3e-3`. The L4 plan drops `3e-4`. The LR stays constant after warmup, as in WSD's stable phase. Compare validation loss at matched tokens and watch for loss spikes. Short runs favor slightly higher rates, so if two are close, pick the lower one.
5. **Save** the results, curves and probe numbers to Drive, with an hourly resumable checkpoint in case Colab disconnects. Only the latest checkpoint is kept, and the notebook purges its own replaced checkpoints from the Drive trash.

| | A100 40 GB | L4 |
| --- | --- | --- |
| Micro-batch (to be confirmed by the probe) | ~32 × 1,024, 16 accumulation steps | ~8–16 × 1,024, 32–64 accumulation steps |
| Expected speed | ~150k tokens/s | ~40k tokens/s |
| Sweep length | 5 rates × 600 steps (315M tokens each) | 4 rates × 400 steps (210M tokens each) |
| Sweep time (estimate) | ~3 h | ~6 h, over two sessions |
| Follow-up | None | Recheck the top two rates for 300 steps each on the Modal H100 (about USD 1–2) |

All speeds and times above were estimates before the run.

**Result.** Colab assigned an L4, so the sweep ran 4 rates × 400 steps at about 41k tokens/s (25% MFU). `2e-3` had the lowest validation loss at every check after warmup: 4.220 at step 400, against 4.307 for `1e-3`, 4.434 for `6e-4` and 4.323 for `3e-3`, which spiked as warmup ended. We picked `2e-3`. The H100 recheck turned out unnecessary: the full run reproduced the sweep's 2e-3 curve (Section 5). [Sweep results](CS336_Colab/README.md#lr-sweep)

## 5. Full training on a Colab L4

The plan was to run the full training on a Modal H100 within a USD 50 budget. We moved it to Colab, for three reasons:

- **Colab handles a multi-day run.** The [training notebook](CS336_Colab/FineWebEdu/CS336_FineWebEdu_Llama124M_Trained_Stage_1.ipynb) saves a checkpoint to Drive every hour, holds an interrupt until the current step finishes, and resumes on whichever GPU the next session gets. A spike guard rolls back loss spikes from in-memory snapshots.
- **The compute units come with the Google AI Pro subscription**, so the whole 10B-token run costs about 120 units instead of about USD 41–50 on Modal.
- **The A100 was never available.** Colab assigned an L4 in every session. The L4 is as cheap per token as the A100, but 3–4 times slower: 40,350 tokens/s against an expected 130–150k. Stage 1 took 20.9 hours instead of about 6, and the whole plan takes about 72 hours instead of about 21.

| Stage | Steps | Tokens | Result |
| --- | ---: | ---: | --- |
| 1. Warmup and stable phase, then a decay branch | 1–4,768, branch 4,769–5,722 | 2.5B + 0.5B | **Done.** Validation loss 3.136, full split **3.126**, in 20.9 h |
| 2. Stable phase continued from step 4,768, then the final decay | 4,769–19,073 | 10.0B | In progress, about 52 h on the L4 |

Results, curves and samples are in [`CS336_Colab/`](CS336_Colab/README.md#fineweb-edu). After Stage 2, both models get a GitHub release and a side-by-side comparison of their generated text.

## 6. After Stage 2: SST-5 fine-tuning and post-training on Colab

Once Stage 2 finishes, a copy of the 124M model is fine-tuned on SST-5 sentiment classification, as in CMU 11-711, with a five-way classification head compared against the majority-class baseline. Running it on both the 2.5B and the 10B model shows what the extra pretraining buys. Post-training (SFT, then DPO) follows. All of it runs in Colab notebooks: SST-5 takes minutes, SFT about 40 minutes per 100M tokens on an L4, and DPO a few hours. At 124M the point is learning the pipeline, not a useful assistant.

## 7. Track 3: a GPT-2 XL-class model on Modal (planned)

Train a model of about 1–1.4B parameters from random weights to a DCLM CORE score of at least **0.2565**, the score of OpenAI's GPT-2 XL, on one Modal 8×H100 node, then fine-tune it into a small chat model. [nanochat](https://github.com/karpathy/nanochat) is the reference for the recipe and the numbers, but **we write our own code**: the model, Muon + AdamW, DDP across 8 GPUs, the data pipeline and the evaluation.

| Phase | What | Cap (USD) |
| --- | --- | ---: |
| 1. Data | A new 32,768-entry tokenizer with chat tokens; about 15B FineWeb-Edu tokens into a Modal Volume | 10 |
| 2. Trainer | Our own multi-GPU trainer, tested on CPU and Colab | 0 |
| 3. Pilot | Speed, memory and resume on 8×H100; prices the full run | 30 |
| 4. Rehearsal | A complete small-depth run to its CORE score | 20 |
| 5. Full run | Depth 24–26 to CORE ≥ 0.2565 | set from the pilot; expected 100–200 |
| 6. Chat | SFT (optionally RL), chat evaluation, export to Colab | 20 |

Each phase starts only when the previous one's gate is met, so at most USD 60 is spent before committing to the full run. An 8×H100 node costs about USD 35 per hour on Modal with CPU and memory; the Starter plan's USD 30 of monthly credits pays for part of each month's work. [The plan](docs/Plan.md) has the recipe, gates and cost model, and the [Runbook](docs/Runbook.md) the checks for every paid run.

Expect fluent text and some general knowledge, not reasoning: that's what GPT-2 XL-level capability means.

## Project docs

- [Plan](docs/Plan.md) · [Runbook](docs/Runbook.md): Track 3 on Modal, its phases, gates, costs and run checks
- [Learning progress](docs/Learning-Progress.md): decisions, checks, experiments and fixes
- [Cloud walker — Modal](docs/cloud-walker-Modal.md): cloud runs, billing, and lessons from 11-768

## Course adaptations

- **Stanford CS336, Assignment 1.** BPE tokenizer, decoder-only Transformer, AdamW, training loop with checkpoints. Track 1 follows it on TinyStories. Track 2 uses FineWeb-Edu in place of OpenWebText. [CS336](https://cs336.stanford.edu/) · [Assignment 1](https://github.com/stanford-cs336/assignment1-basics)
- **CMU 11-711, Build Your Own Llama.** Llama 2 blocks, sampling with temperature and top-p, SST-5 fine-tuning. We pretrain our own checkpoint on FineWeb-Edu instead of loading the course's TinyStories reference model, and use our own tokenizer throughout.
- **Stanford CS336, Assignments 2–5 (Track 3, planned).** Systems: DDP across 8 GPUs, profiling and MFU. Scaling and data: the depth dial, the token budget and a larger data pipeline. Alignment: SFT, and optionally RL. [Plan](docs/Plan.md#course-work-and-adaptations)
