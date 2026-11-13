# Stanford CS336 Llama Adaptation

Train one **123,489,024-parameter Llama-style language model from random weights** on TinyStories, then fine-tune a copy on the five-class SST-5 sentiment task. Codespaces runs the commands. Modal runs data preparation, pretraining, sampling, and fine-tuning. The pilot is the first segment of the same pretraining run.

## Start from GitHub Codespaces

Open **Code → Codespaces → Create codespace on main**. The devcontainer installs Python 3.12, uv, the Modal CLI, and CPU PyTorch from the lockfile. GitHub Actions runs the CPU tests and checks the Modal CLI without allocating GPUs.

```bash
uv run --locked --extra cpu python -m pytest -q
uv run --locked --extra cpu modal token new
```

Authenticate in the browser when prompted and select the intended Modal workspace. If you already use Codespaces secrets named `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, grant this repository access to them in GitHub and restart the Codespace. Use the same Modal workspace and environment for every command.

Define this shorthand in each new terminal:

```bash
m() { uv run --locked --extra cpu modal "$@"; }
```

## Prepare the dataset and run the pilot

```bash
m run --detach modal_app.py::prepare
```

Save the **app ID** (`ap-…`) printed by Modal. You can close Codespaces once the app has started. Check progress with `m app list` or `m app logs ap-REPLACE_ME`. Wait for the preparation app to finish and confirm that the dataset manifest exists:

```bash
m volume get cs336-llama-adaptation-artifacts tinystories-v2/manifest.json -
```

Preparation uses CPU compute. It downloads the pinned TinyStoriesV2 GPT-4 train and validation files, fits byte-level BPE on the first 100,000 training stories, inserts end-of-story tokens, and writes packed train and validation streams to a Modal Volume. The tokenizer, source revision, and token counts are recorded in the manifest. A completed preparation can be run again and will reuse its artifacts.

```bash
m run --detach modal_app.py::pretrain --run-id llama124m --target-step 100 --minutes 20
```

This starts one H100. The default batch is 8 sequences × 8 accumulation passes × 256 tokens = **16,384 tokens per optimizer step**. Step 100 is **1.64M processed tokens**. The logs show step, cumulative tokens, training loss, throughput, and checkpoint validation loss. The first log also shows the randomly initialized model's validation loss.

```bash
m app logs ap-REPLACE_WITH_PILOT_ID
m volume ls cs336-llama-adaptation-artifacts runs/llama124m
```

## Resume pretraining

Once the previous app has finished, submit the next segment with the **same run ID**. The target step is cumulative.

```bash
m run --detach modal_app.py::pretrain --run-id llama124m --target-step 1000 --minutes 90
```

Look for `resumed step=100` if the pilot reached its target. If it reached the time limit first, expect the last saved step. The run saves `latest.pt` every 500 steps and at each planned stop. It saves model weights, optimizer state, batch sampling RNG state, validation loss, and the exact run settings. Small `metrics-*.json` files record validation loss, processed tokens, and peak allocated GPU memory. A new invocation resumes from the last committed checkpoint. Keep schedule, warmup, micro-batch, accumulation, and peak learning rate identical across segments. You can change the target step, runtime, and checkpoint interval. The default cosine schedule ends at step 15,000, or 245.76M processed tokens.

The example below targets step 5,000 (81.92M cumulative tokens). Set the target and runtime using your measured throughput and remaining budget. If an app reaches its time limit before the target, launch another segment with the same target.

```bash
m run --detach modal_app.py::pretrain --run-id llama124m --target-step 5000 --minutes 360
```

Estimate **GPU hours ≈ remaining tokens ÷ measured tokens/sec ÷ 3,600**. Include startup, checkpointing, evaluation, and CPU/RAM when setting a budget. Modal lists the H100 at approximately **US$3.95 per GPU-hour**. Use **US$200 as the total project cap**, with a working allocation of $20 for preparation and pilot, $130 for pretraining, $20 for SST-5, and $30 in reserve. Check workspace billing before each new segment. Modal's workspace limits cover all projects in the current billing cycle, and the per-invocation runtime applies to each execution attempt. Stop a running app from the Modal dashboard or with `m app stop ap-REPLACE_ME`. [Modal pricing](https://modal.com/pricing) · [Workspace budgets](https://modal.com/docs/guide/budgets)

Each run allows one pretraining writer. If a hard cancellation leaves a stale writer lock, inspect it with `m dict items cs336-llama-adaptation-locks`. After confirming **all preparation, pretraining, and sentiment apps for this project have stopped**, run `m dict clear cs336-llama-adaptation-locks` and resume the run. The lock dictionary contains only writer IDs.

## Sample stories and fine-tune sentiment

Run sampling after a pretraining segment completes. It runs on CPU and prints and saves a JSON file with three prompts, their continuations, the checkpoint step, and sampling settings. Use the same settings at two checkpoint steps to see what changed.

```bash
m run modal_app.py::sample --run-id llama124m
m run modal_app.py::sample --run-id llama124m --prompt "Nora found a key under the garden bench."
```

Once your continuations show a coherent thread, fine-tune a copy of the checkpoint on SST-5. It uses a five-way classification head over the last unpadded token and updates the backbone and the head. The logs and metrics report dev accuracy by epoch and the dev accuracy of the training-set majority class.

```bash
m run --detach modal_app.py::sentiment --run-id llama124m --experiment sst5
m volume get cs336-llama-adaptation-artifacts runs/llama124m/sentiment/sst5/metrics.json -
```

Run a second seed if time allows. Both commands load the same pretraining checkpoint if you launch them before the next pretraining segment. Sentiment weights go to `sentiment/<experiment>/best.pt` inside the run folder, leaving the base checkpoint available for generation and resume.

```bash
m run --detach modal_app.py::sentiment --run-id llama124m --experiment sst5-seed337 --seed 337
```

## One week, about an hour a day

| Day | Work in Codespaces |
| --- | --- |
| 1 | Open the Codespace, run tests, authenticate Modal, and launch CPU data prep. |
| 2 | Check the manifest, launch the 100-step pilot, and read the initial and final validation losses. |
| 3 | Resume to step 1,000. Check throughput and estimate the next segment from the observed cost. |
| 4 | Sample stories and launch the next segment within the remaining budget. |
| 5 | Compare samples and validation loss. Continue pretraining if useful. |
| 6 | Run SST-5 fine-tuning and collect dev accuracy and the majority-class baseline. |
| 7 | Record steps, tokens, validation loss, story samples, SST-5 results, and Modal spend. |

## Course adaptations

- **Stanford CS336.** Build the decoder-only Transformer and training pipeline, train a new BPE tokenizer on TinyStories, and measure training throughput. This one-week version uses Hugging Face Tokenizers for BPE training and PyTorch for AdamW and scaled dot-product attention. [Stanford CS336](https://cs336.stanford.edu/) · [Assignment 1 materials](https://github.com/stanford-cs336/assignment1-basics)
- **CMU 11-711 Fall 2026.** Implement a Llama-style architecture and apply the generation, validation, and SST-5 exercises to your own checkpoint. Use the project tokenizer and raw TinyStories files throughout. The CMU reference checkpoint uses its own tokenizer. This repo evaluates the pretrained model's generations and SST-5 fine-tuning using the project's vocabulary. [CMU Homework 1](https://cmu-anlp.org/assignments/01)
- **Architecture.** 12 decoder blocks, width 768, 12 query heads, 4 key/value heads, SwiGLU width 3,328, 16,384 vocabulary entries, and context length 256. RMSNorm, RoPE, bias-free projections, and tied input/output embeddings.
