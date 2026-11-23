# Project plan: a GPT-2 XL-class model on Modal

Train a GPT-2 XL-class language model from random weights on one Modal 8×H100 node, then fine-tune it into a small chat model. The target is **GPT-2 XL's capability, not its exact recipe**: a DCLM CORE score of at least **0.2565**, the score of OpenAI's 1.5B GPT-2, reached with a modern training recipe for a small fraction of the classic cost.

This is Track 3. Tracks 1 and 2 (TinyStories and the 124M FineWeb-Edu model) run on Colab. SST-5 fine-tuning and post-training for the 124M model also stay on Colab, after Track 2's Stage 2.

**Status:** planned. No Track 3 code exists yet. This plan comes first, and the code follows one phase at a time, each in its own pull request.

## Why the modern recipe, not a classic GPT-2 XL reproduction

| | Classic GPT-2 XL | Modern recipe (this plan) |
| --- | --- | --- |
| Recipe | Track 2's Llama + AdamW, scaled to 48 layers | nanochat-style: Muon for the weight matrices, modern block details, a depth-driven configuration |
| Tokens | ~30B (20 per parameter) | ~8–15B (8–10.5 per parameter) |
| Reference | llm.c: 32B tokens, 24 h on 8×H100, USD 672 at the rental price it used | nanochat: GPT-2 XL's CORE in 1.65–3 h on 8×H100, about USD 48 at USD 3 per GPU-hour |
| Our estimate on Modal | USD 1,000–1,400 per attempt | **USD 100–200 per attempt** (our own code will be slower than nanochat's) |
| End result | A base model at GPT-2 XL level | A base model at GPT-2 XL level, then SFT into a chat model |

Both teach multi-GPU training, a larger data pipeline and the operations of a long run. The modern recipe also teaches the optimizer and architecture changes behind recent small-model progress, and it fits a budget that can be saved up for in weeks rather than months. The classic cost table stays in the [README](../README.md#what-a-classic-reproduction-would-cost-on-modal) for reference.

**Set expectations.** At this scale, "GPT-2 XL capability" means fluent paragraphs and some general knowledge, not reasoning. nanochat describes its GPT-2-grade chat model as a bit like talking to a kindergartener. The 1.5B-class models that feel capable today were trained on trillions of tokens.

## Recipe

[nanochat](https://github.com/karpathy/nanochat) (MIT) is the reference for the recipe and the numbers. **We write our own implementation**, one piece at a time, and use nanochat to check our choices and results, not as a dependency.

| Part | Plan | Notes |
| --- | --- | --- |
| Size | One depth dial sets width, heads and the training horizon. Target depth 24–26 | GPT-2 capability arrives around depth 24–26 in nanochat. That's about 1.0–1.4B parameters, depending on details such as value embeddings |
| Blocks | RoPE, RMSNorm with QK-norm, ReLU² MLP, no biases, untied embeddings, logit soft-capping | The parts nanochat combines. We add them one at a time, each with a test |
| Optimizer | Muon for the 2-D weight matrices, AdamW for embeddings and scalars | Learning rates scale with depth and width, as in nanochat |
| Precision | bf16 compute with FP32 master weights; fp8 later, if the pilot shows it pays | nanochat's leaderboard used fp8 for part of its gain |
| Context and batch | 2,048 tokens, about 0.5–1M tokens per step | nanochat moved from 0.5M to 1M tokens per step |
| Tokenizer | A new 32,768-entry BPE, with chat special tokens reserved from the start | Track 2's 32,000-entry tokenizer has no chat tokens, and adding them later means resizing the embeddings |
| Data | FineWeb-Edu `sample/100BT`, about 15B tokens (≈30 GB as `uint16`) | Keeps continuity with Track 2. nanochat later switched to NVIDIA ClimbMix, which reached GPT-2 sooner; we can test that in the rehearsal |
| Evaluation | CORE (the DCLM suite, GPT-2 XL = 0.2565), validation bits per byte, HellaSwag | Bits per byte doesn't depend on the tokenizer, so it also compares with Track 2 |
| Chat | SFT on conversation data such as SmolTalk, with the loss only on assistant tokens. RL on GSM8K optional | Chat evaluation: ARC, MMLU, GSM8K |

## Phases, gates and spending caps

Each phase has a spending cap and a gate: the next phase starts only when the gate is met. Phases 1–4 together cap at **USD 60**, spent before committing to the full run.

| Phase | What | Where | Gate to move on | Cap (USD) |
| --- | --- | --- | --- | ---: |
| 0. Setup | Modal workspace, budget alerts, Volume, image | Modal | Budget limit set below the next phase's cap | 0 |
| 1. Data | Train the tokenizer; tokenize about 15B tokens into the Volume with a manifest and checksums | Modal CPUs | Manifest checks pass; token count and bytes per token recorded | 10 |
| 2. Trainer | Model, Muon + AdamW, DDP, a deterministic data loader sharded by rank, resumable checkpoints, the spike guard, CORE and bits-per-byte evaluation | Codespaces, CPU tests, Colab | Tests pass. Two CPU processes (DDP with the gloo backend) give the same losses as one process with gradient accumulation. A small model trains on Colab | 0 |
| 3. Pilot | 8×H100: a 10-minute smoke run at depth 12, then about 200 steps at the target depth; save and resume across function calls | Modal | Measured tokens/s, MFU and memory; the projected full-run cost fits the Phase 5 cap | 30 |
| 4. Rehearsal | A complete run at depth 12, the whole schedule, ending with its CORE score | Modal 8×H100 | Loss curve and CORE in line with nanochat's small models; the schedule and learning rates behave as planned | 20 |
| 5. Full run | Depth 24–26 to CORE ≥ 0.2565 | Modal 8×H100 | — | Set from the pilot; expected 100–200 |
| 6. Chat | SFT (and optionally RL), chat evaluation, FP16 export; chat on Colab | Modal, then Colab | — | 20 |

**Expected total: about USD 180–290**, of which up to USD 60 is spent on phases 1–4 before committing to the full run. Modal's Starter plan includes USD 30 of compute credits each month, which covers a phase like the data work in one month. Spreading the early phases over a few months lets the credits pay for much of them.

If a gate fails, fix the problem and repeat the phase, or shrink the plan: a smaller depth, fewer tokens, or fp8. Don't start Phase 5 on hope.

## Cost model

- **8×H100 node.** USD 3.95 per H100-hour × 8 = USD 31.60/h for the GPUs. CPU cores and memory are billed on top, about USD 2–4/h for a training container. **Plan with about USD 35 per node-hour.**
- **Full run.** nanochat reaches GPT-2 XL's CORE in 1.65–3 h on 8×H100. Allowing for our code being slower, plan **3–6 h, so USD 100–200**. The pilot replaces this estimate with the measured cost per billion tokens.
- **Checkpoints.** A 1–1.4B-parameter model with FP32 master weights and optimizer state takes roughly 11–17 GB per checkpoint. Keep one rolling checkpoint plus milestone copies. The first 1 TiB of Volume storage each month is included.
- **Time limits.** One Modal function call runs at most 24 hours, so training is resumable by design: retries restart from the last checkpoint. A stall guard stops a job that has stopped making progress.
- **GPU type.** Modal may upgrade `H100` to `H200` at no extra cost. Benchmarks in the pilot pin the type (`H100!`) so measurements are comparable.

Check [Modal pricing](https://modal.com/pricing) before each phase; the numbers above are from November 2026.

## Course work and adaptations

| Course material | Where it shows up |
| --- | --- |
| Stanford CS336 A1, Basics | Tokenizer training, the decoder-only Transformer, the training loop (Tracks 1–3) |
| Stanford CS336 A2, Systems | Track 3: DDP across 8 GPUs, sharded optimizer state, profiling, MFU and memory measurement |
| Stanford CS336 A3–A4, Scaling and data | Track 3: the depth dial and tokens-per-parameter choice, the data pipeline and its manifest |
| Stanford CS336 A5, Alignment | Track 3 Phase 6: SFT, and optionally RL on GSM8K |
| CMU 11-711, Build Your Own Llama | Llama blocks, sampling, SST-5 fine-tuning (Track 2's model, on Colab) |

[Stanford CS336](https://cs336.stanford.edu/) · [nanochat](https://github.com/karpathy/nanochat) · [llm.c GPT-2 1.5B](https://github.com/karpathy/llm.c/discussions/677) · [Runbook](Runbook.md) · [Learning progress](Learning-Progress.md) · [Modal journal](cloud-walker-Modal.md)
