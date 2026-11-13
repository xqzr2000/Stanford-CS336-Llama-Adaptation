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
| The schedule length is stored in the checkpoint, so a pilot inside the main run fixes it before throughput is known. | The pilot uses its own run ID. The main run starts after the schedule is chosen from the [cost estimate](Plan.md#cost-estimate). |

The trainer also reads the token files into memory, so random batch windows do not depend on Volume read speed.

**Still open.** The stall guard, the CUDA image, and H100 throughput are untested until the pilot. The cost table uses assumed throughput.

## Entry template

Append a dated entry above this template.

- **Goal and setup** — Commit, dataset revision, checkpoint step, and the question for the session.
- **Observed** — Relevant commands or metrics, including units, checkpoint steps, sample prompts, or exact error text.
- **Diagnosis and fix** — What was tried, why it was chosen, and what verified the result. Mark an open hypothesis as such.
- **What we learned** — Explain one model or training concept in terms of the run.
- **Next** — The next concrete action and the evidence to collect.
