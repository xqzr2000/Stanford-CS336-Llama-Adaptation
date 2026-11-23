# Learning progress

A chronological record of what we built, what we observed, and what we changed. Add each session at the bottom. Use the [plan](Plan.md) for targets, the [runbook](Runbook.md) for commands, and the [Modal journal](cloud-walker-Modal.md) for infrastructure and billing.

## Day 0 · November 13, 2026 · Decide what to train

We chose to combine the shared core of Stanford CS336 and CMU 11-711 in one codebase. The first idea had a small TinyStories baseline and a second model at GPT-2 Small scale. We settled on one **123,489,024-parameter Llama-style model** trained from random weights. The short pilot now produces the first checkpoint for the longer run. GPT-2 Small supplied the approximate parameter target. The architecture uses RoPE, RMSNorm, SwiGLU, grouped-query attention, and tied embeddings.

Our CMU 11-768 project updated adapters on an existing Qwen checkpoint using reinforcement learning. Here, the pretraining objective is to predict the next token in TinyStories. We will look for falling held-out loss and increasingly coherent continuations, then apply a five-class sentiment head to a copy of the pretrained checkpoint and fine-tune it on SST-5.

With one hour per day, we chose to learn the model and training workflow through one end-to-end run. We fit our own byte-level BPE vocabulary using Hugging Face Tokenizers and use PyTorch implementations of AdamW and scaled dot-product attention. Those choices let us focus on model behavior, training, and evaluation. The [plan](Plan.md#course-work-and-adaptations) maps each adaptation to the course work.

## Day 0 · November 13, 2026 · Implement and verify the pipeline

| Check | Observed result |
| --- | --- |
| TinyStories preparation | Fit a 16,384-token byte-level BPE tokenizer on the training split and processed the full pinned files locally. The resulting manifest recorded **533,882,526 training tokens** and **5,391,750 validation tokens**. |
| Model size and training | The full-size model completed two CPU optimizer steps on a small TinyStories slice. It saved after the first step, resumed from disk, and generated text from the resumed checkpoint. This was a functional smoke test. |
| Checkpoint reproducibility | With the small test configuration, the resumed and uninterrupted CPU runs produced identical model weights and validation loss. The test also rejected incompatible settings. |
| Causal attention and padding | The unit tests checked that extending a sequence leaves earlier outputs unchanged and that right padding leaves the classifier output unchanged. |
| SST-5 | Loaded **8,544 training** and **1,101 development** examples from the pinned dataset. A small fixture exercised classifier fine-tuning, the majority baseline, and metrics and checkpoint output. |
| GitHub Actions | All three CPU tests passed and the Modal pretraining CLI loaded. [CI run](https://github.com/xqzr2000/Stanford-CS336-Llama-Adaptation/actions/runs/43489356144) |

**Error and fix.** The first parameter-count test expected 123,487,488. The model reported 123,489,024. Recounting the implemented weights and confirming the count with a real CPU forward and backward pass resolved the mismatch. We corrected the test's expected value.

**What the checks established.** The data pipeline builds a tokenizer and packed token streams, and the trainer can pick up from a saved optimizer and batch-sampling state. The local token counts describe the dataset, while processed tokens in a training log count the tokens the model has seen during optimization. The first H100 pilot, sustained GPU throughput, story coherence, and full SST-5 accuracy are pending.

**Next session.** Prepare the data on Modal, confirm the Volume manifest, then run the first 100-step H100 pilot. Save the step-0 and final held-out loss and check that the checkpoint is visible from a fresh CLI session. [Runbook](Runbook.md)

## Day 0 · November 13, 2026 · Review before spending on Modal

A read-through of the code before the first GPU job found no crash-level errors and four cost risks. All four are changed in code. None of the changes has run on Modal yet.

| Risk | Change |
| --- | --- |
| A stuck job could bill an H100 for 24 hours, because the runtime limit is checked only after a completed step. | The function timeout is 6 hours and one invocation accepts at most 330 minutes. A stall guard fails the job after 15 minutes without a completed step or checkpoint. |
| Throughput and training loss were only in the logs, which the Starter plan keeps for one day. | Each `metrics-*.json` now records training loss, `train_tokens_per_second`, and validation and save time. |
| The default batch ran 8 sequences per pass with 8 accumulation passes, which uses a small part of the H100 per pass. | The default is 64 sequences in one pass, the same 16,384 tokens per step. Validation now covers 512 windows. |
| The schedule length is stored in the checkpoint, so a pilot inside the main run fixes it before throughput is known. | The pilot uses its own run ID. The main run starts after the schedule is chosen from the [cost estimate](https://github.com/xqzr2000/Stanford-CS336-Llama-Adaptation/blob/75bc2dc/docs/Plan.md#cost-estimate). |

The trainer also reads the token files into memory, so random batch windows do not depend on Volume read speed.

**Still open.** The stall guard, the CUDA image, and H100 throughput are untested until the pilot. The cost table uses assumed throughput.

## Day 0 · November 13, 2026 · Switch pretraining data to FineWeb-Edu

We replaced TinyStories with the pinned `sample/10BT` subset of FineWeb-Edu, a set of educational web pages filtered from Common Crawl. The local checks recorded above used TinyStories and no longer describe the prepared data.

| Decision | Reason |
| --- | --- |
| Train on shards 000–002, about 2.5B tokens, and validate on the first 5M tokens of shard 013. | Three shards give about one pass at roughly 20 tokens per parameter. FineWeb-Edu has no validation split, so a shard that training never reads serves as one. |
| Download one shard at a time to local disk and delete it after encoding. | The Volume keeps only the 5 GB of token streams. |
| Keep the 16,384-entry vocabulary and 256-token context. | The parameter count stays at 123,489,024 and the earlier tests and estimates still apply. Both are small for web text: a larger vocabulary compresses it better and a longer context keeps more of each page. |
| Raise the default schedule to 150,000 steps and the checkpoint interval to 2,000. Raise pretraining memory to 24 GiB. | The schedule matches the prepared data. The interval keeps checkpoints to 75 over the run. The trainer holds the training tokens in memory. |
| Use educational prompts for sampling. | Story prompts no longer match the training data. |

**What to expect.** Web text is harder to model than TinyStories, so validation loss will stay higher and samples will be less coherent at the same size. SST-5 may benefit, because web text covers more opinions and reviews than children's stories.

**Still open.** The FineWeb-Edu revision is pinned from the Hugging Face file listing, and the real download and tokenization have not run yet. The CPU tests cover Parquet reading, the validation cap, shard cleanup, and the manifest checks on a small synthetic dataset. FineWeb-Edu contains near-duplicate pages across crawls, so some validation documents may resemble training documents.

## Day 0 · November 14, 2026 · Colab companion notebook

The walkthrough notebook moved into the repo at `notebooks/` and now covers everything except H100 pretraining. It clones the repo at a chosen revision, `main` by default, and calls the same functions Modal runs.

| Chapter | What it runs |
| --- | --- |
| 1 · From token IDs to logits | The model checked against the math for RMSNorm, RoPE, grouped-query attention, SwiGLU, and the loss. |
| 2 · FineWeb-Edu on a slice | `prepare_data` on the first 2,000 documents of the pinned shards, with a 4,096-entry tokenizer. |
| 3 · Training, small and large | The learning-rate schedule, `pretrain_model` on the slice with a small model including a resume, and the H100 run's metrics and cost. |
| 4 · Host the final model | The exported weights, the project's sampling settings, and validation loss recomputed on the T4. |
| 5 · SST-5 on the T4 | Fine-tuning a copy of the final model, or of the chapter-3 model in practice mode. |

| Change | Reason |
| --- | --- |
| `precision` uses BF16 autocast only on GPUs with compute capability 8.0 or higher. | The T4 has no native BF16, so the repo now runs in FP32 there. The H100 path is unchanged. |
| `load_for_eval` checks the tokenizer hash directly. | It no longer needs `train.bin` and the manifest, which Colab doesn't have. |
| New `export` Modal job writes a `colab` folder with FP16 weights. | The weights take about 247 MB instead of the 1.48 GB checkpoint, which also holds AdamW state. |
| SST-5 is planned on Colab's T4. | The job is small, and the Modal `sentiment` job stays as a fallback. |

**Checked.** The whole notebook ran on CPU against this branch with a real full-size checkpoint and its export, synthetic Parquet shards in place of the Hugging Face download, and small synthetic SST-5 files. The practice model's validation loss fell from 8.33 to 5.31 over 300 steps, with a stop and resume at step 150.

**Still open.** Streaming the slice from Hugging Face, the T4 itself, Drive mounting, and downloading the real SST-5 files have not run yet.

## Day 0 · November 14, 2026 · CS336_Colab track for TinyStories

We added `CS336_Colab/`, a Colab track that follows the original CS336 Assignment 1: a 22.7M-parameter Transformer trained on TinyStories on a T4, with its own tokenizer, model, and AdamW in one notebook. It follows the layout of `Mario_Colab` in Mario_Ops: the notebook, a README, and the trained model committed after training.

| Decision | Reason |
| --- | --- |
| Sweep the peak learning rate on the 40m profile (3e-4, 1e-3, 3e-3), then train the full profile at the best rate. | The learning rate matters most for the handout's 2.00 and 1.45 references, and each 40m run is short on a T4. |
| Keep hand-written attention and add a `FAST_ATTENTION` switch with an equivalence check. | Writing attention is part of the assignment. The fused kernel avoids storing the 256 × 256 score matrices, the slowest part on a T4. |
| Save every 1,000 iterations on the full run instead of 250. | A full checkpoint is about 272 MB, and writing one to Drive takes several seconds. |
| Export FP16 weights with the tokenizer, about 46 MB. | The model fits in git under GitHub's 50 MB warning, and `GENERATE_FROM = "github"` can load it without Drive. |
| Score the whole validation split when a run finishes. | The sampled windows during training are for comparing checkpoints. The final number covers every validation token. |
| Allow `*.pt` in `CS336_Colab/`. | `.gitignore` excludes `*.pt` everywhere else. |

**Checked.** Every mode ran end to end on CPU with a scaled-down model and a small stand-in corpus: a sweep including a deliberately divergent rate, a rerun that skipped finished and diverged rates, a full run paused at iteration 20 and resumed with fast attention, generation from Drive and from GitHub, and packaging. A full-size FP16 export measured 45.7 MB.

**Still open.** The T4 itself, Drive, the TinyStories download, and the real throughput have not run yet.

## November 17–18, 2026 · Completed the TinyStories run and published the model

We ran a two-round learning-rate sweep on a Colab T4. Each run processed 40.96M tokens. Of the five rates we tried (`3e-4`, `1e-3`, `3e-3`, `5e-3`, `7e-3`), `3e-3` gave the lowest full-split validation loss, **1.6544**. We started the full run from random weights with that peak rate and a schedule sized for 40,000 steps.

The sweep's frequent checkpoints took up about 10 GB in Drive. For the full run, we switched to hourly saves with two retained checkpoints. The notebook resumes from the newest readable checkpoint and can fall back to the previous one. It also reuses the saved BPE tokenizer and encoded token files, so reopening Colab with the same run settings skips data preparation and the sweep.

The full run processed **327.68M tokens** and reached **1.4466** on the fixed 512-window validation sample and **1.4099** on the full split, for a perplexity of **4.10**. In the saved blue-kite comparison, the sweep sample drifts between characters while the full-run sample keeps the robot and the girl in a story with a clear problem and resolution. [Trained notebook, Section 12](../CS336_Colab/TinyStories/CS336_TinyStories_Trained.ipynb).

We committed the [metrics](../CS336_Colab/TinyStories/runs/metrics.json) and [learning curves](../CS336_Colab/TinyStories/runs/learning_curves.png) and published the FP16 export as [`model.pt` on the `tinystories-v1` release](https://github.com/xqzr2000/Stanford-CS336-Llama-Adaptation/releases/tag/tinystories-v1). The export includes the tokenizer and model configuration for CPU generation.

## November 19–22, 2026 · Track 2 Stage 1 on Colab, and a new plan for Modal

**Track 2 moved to Colab.** The LR sweep (4 rates × 400 steps on an L4) picked a peak rate of `2e-3`: 4.220 at step 400, while `3e-3` spiked as warmup ended. The full run then trained on Colab instead of a Modal H100. It saves a checkpoint to Drive every hour, resumes on any GPU, and the compute units come with the Google AI Pro subscription. Colab never assigned an A100, so it ran on an L4: about as cheap per token, but 3–4 times slower.

**Stage 1 is done.** 2.5B tokens at `2e-3`, then a 0.5B-token linear decay branched from the step-4,768 checkpoint, gave a validation loss of **3.136** (full split **3.126**) in 20.9 hours. Stage 2 continues the main run to 10B tokens. [Results](../CS336_Colab/README.md#fineweb-edu).

| Decision | Reason |
| --- | --- |
| Sample training windows without replacement, from a seeded order per epoch. | Over one pass of the data, sampling with replacement leaves about a third of the tokens unseen. |
| Keep a permanent pre-decay checkpoint and branch the 2.5B model's decay from it. | The main run continues at a constant rate, so the decay never touches it. |
| Plan Track 3 as a GPT-2 XL-class model with a modern recipe on Modal 8×H100. | Reaching GPT-2 XL's CORE takes about USD 100–200 this way, against USD 1,000–1,400 for a classic 30B-token reproduction. See the [plan](Plan.md). |
| Delete the old 124M Modal code (`llama_lab/`, `modal_app.py`, its tests, the walkthrough notebook) and its CI. | Track 2 doesn't use it. Track 3 needs a new multi-GPU trainer, written phase by phase. Git history keeps the old code. |
| Keep SST-5 fine-tuning and post-training for the 124M model on Colab, after Stage 2. | They're small jobs at this size and fit the same notebook workflow. |

**What we learned.** The decay matters as much as the length of the run. On the same batches at step 5,000, the decaying branch was 0.027 nats ahead of the run still at `2e-3`, and it gained 0.138 nats over the whole decay. A WSD schedule gets a finished model at any point of the run for the price of a short decay.

**Next.** Finish Stage 2 on the L4, in sessions of about 10 hours. Start Track 3 Phase 1 (tokenizer and data) when the plan is merged.

## Entry template

Append a dated entry above this template.

- **Goal and setup** — Commit, dataset revision, checkpoint step, and the question for the session.
- **Observed** — Relevant commands or metrics, including units, checkpoint steps, sample prompts, or exact error text.
- **Diagnosis and fix** — What was tried, why it was chosen, and what verified the result. Mark an open hypothesis as such.
- **What we learned** — Explain one model or training concept in terms of the run.
- **Next** — The next concrete action and the evidence to collect.
