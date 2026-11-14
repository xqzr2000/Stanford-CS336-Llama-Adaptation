# Stanford CS336 Llama Adaptation

Train one **123,489,024-parameter Llama-style language model from random weights** on FineWeb-Edu, a corpus of educational web pages, then fine-tune a copy on the five-class SST-5 sentiment task. Codespaces runs the commands. Modal runs data preparation, pretraining, sampling, and fine-tuning. A 100-step pilot under its own run ID measures throughput before the main run starts.

## Project docs

- [Plan](docs/Plan.md) covers the one-week schedule, course adaptations, cost estimate, and US$200 budget.
- [Runbook](docs/Runbook.md) has the Codespaces and Modal commands, checkpoints to verify, and recovery steps.
- [Learning progress](docs/Learning-Progress.md) records decisions, local checks, experiments, and fixes.
- [Cloud walker — Modal](docs/cloud-walker-Modal.md) records cloud runs, billing, and lessons from 11-768.

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
m volume get cs336-llama-adaptation-artifacts fineweb-edu-10bt/manifest.json -
```

Preparation uses CPU compute. It reads the pinned `sample/10BT` subset of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) from Hugging Face. It downloads one 2.15 GB Parquet shard at a time to the container's local disk, fits byte-level BPE on the first 100,000 documents, adds an end-of-text token after each document, appends the tokens to the packed stream on the Modal Volume, and deletes the shard. By default it uses shards 000–002 for training, about 2.5B tokens and 5 GB on the Volume. Validation uses the first 5M tokens of shard 013, which training never reads. `--train-shards` accepts 1–6. The tokenizer, revision, shard list, and token counts are recorded in the manifest. A completed preparation can be run again and will reuse its artifacts. To change the shard count afterwards, remove the dataset folder with `m volume rm -r cs336-llama-adaptation-artifacts fineweb-edu-10bt` first.

The download is anonymous. Hugging Face limits anonymous downloads per IP address, and Modal containers can share outbound addresses, so the job retries rate-limit responses. If preparation still fails with HTTP 429, create a Modal Secret holding `HF_TOKEN`, attach it to `prepare` in `modal_app.py`, and run it again.

```bash
m run --detach modal_app.py::pretrain --run-id pilot --target-step 100 --minutes 20
```

This starts one H100. The default batch is 64 sequences × 256 tokens = **16,384 tokens per optimizer step**, with one pass per step. Step 100 is **1.64M processed tokens**. The logs show step, cumulative tokens, training loss, throughput, and checkpoint validation loss. The first log also shows the randomly initialized model's validation loss. Each `metrics-*.json` file also records the latest training loss, `train_tokens_per_second`, and the time spent on validation and saving, so the measurements outlive Modal's log retention (one day on the Starter plan).

```bash
m app logs ap-REPLACE_WITH_PILOT_ID
m volume get cs336-llama-adaptation-artifacts runs/pilot/metrics-000100.json -
```

## Start and resume the main run

Use the pilot's throughput and the [cost estimate](docs/Plan.md#cost-estimate) to choose the length of the main run **before its first step**. The checkpoint stores the schedule length, warmup, batch settings, and peak learning rate, and a segment with different values is rejected. The default schedule is 150,000 steps, 2.46B processed tokens, about one pass over the default training data and about 20 tokens per parameter. For a different length, add the same `--schedule-steps N` to every segment below, starting with the first.

Start the main run under a new run ID. Once an app has finished, submit the next segment with the **same run ID**. The target step is cumulative.

```bash
m run --detach modal_app.py::pretrain --run-id llama124m --target-step 1000 --minutes 90
```

The first segment starts from random weights. On later segments, look for `resumed step=...` with the last saved step. The run saves `latest.pt` every 2,000 steps and at each planned stop. It saves model weights, optimizer state, batch sampling RNG state, validation loss, and the exact run settings. Small `metrics-*.json` files record validation loss, processed tokens, and peak allocated GPU memory. A new invocation resumes from the last committed checkpoint. Keep schedule, warmup, micro-batch, accumulation, and peak learning rate identical across segments. You can change the target step, runtime, and checkpoint interval. The default cosine schedule ends at step 150,000, or 2.46B processed tokens.

The example below targets step 5,000 (81.92M cumulative tokens). Set the target and runtime using your measured throughput and remaining budget. If an app reaches its time limit before the target, launch another segment with the same target.

```bash
m run --detach modal_app.py::pretrain --run-id llama124m --target-step 5000 --minutes 120
```

One invocation accepts at most 330 minutes, and Modal stops the function 30 minutes after that. The job also fails itself if no optimizer step or checkpoint completes for 15 minutes. In both cases the last committed checkpoint remains, and the same command resumes from it.

Estimate **GPU hours ≈ remaining tokens ÷ measured tokens/sec ÷ 3,600**. Include startup, checkpointing, evaluation, and CPU/RAM when setting a budget. Modal lists the H100 at approximately **US$3.95 per GPU-hour**. Use **US$200 as the total project cap**, with a working allocation of $20 for preparation and pilot, $130 for pretraining, $20 for SST-5, and $30 in reserve. Check workspace billing before each new segment. Modal's workspace limits cover all projects in the current billing cycle, and the per-invocation runtime applies to each execution attempt. Stop a running app from the Modal dashboard or with `m app stop ap-REPLACE_ME`. [Modal pricing](https://modal.com/pricing) · [Workspace budgets](https://modal.com/docs/guide/budgets)

Each run allows one pretraining writer. If a hard cancellation leaves a stale writer lock, inspect it with `m dict items cs336-llama-adaptation-locks`. After confirming **all preparation, pretraining, and sentiment apps for this project have stopped**, run `m dict clear cs336-llama-adaptation-locks` and resume the run. The lock dictionary contains only writer IDs.

## Sample stories and fine-tune sentiment

Run sampling after a pretraining segment completes. It runs on CPU and prints and saves a JSON file with three educational prompts, their continuations, the checkpoint step, and sampling settings. Use the same settings at two checkpoint steps to see what changed.

```bash
m run modal_app.py::sample --run-id llama124m
m run modal_app.py::sample --run-id llama124m --prompt "Nora found a key under the garden bench."
```

Once continuations stay on the prompt's topic in grammatical sentences, fine-tune a copy of the checkpoint on SST-5. It uses a five-way classification head over the last unpadded token and updates the backbone and the head. The logs and metrics report dev accuracy by epoch and the dev accuracy of the training-set majority class.

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
| 2 | Check the manifest, launch the 100-step pilot, and read the validation losses and throughput. |
| 3 | Choose the schedule length from the pilot's throughput. Start the main run and train to step 1,000. |
| 4 | Sample stories and launch the next segment within the remaining budget. |
| 5 | Compare samples and validation loss. Continue pretraining if useful. |
| 6 | Run SST-5 fine-tuning and collect dev accuracy and the majority-class baseline. |
| 7 | Record steps, tokens, validation loss, story samples, SST-5 results, and Modal spend. |

## Course adaptations

- **Stanford CS336.** Build the decoder-only Transformer and training pipeline, train a new BPE tokenizer on web text, and measure training throughput. FineWeb-Edu stands in for the assignment's OpenWebText. This one-week version uses Hugging Face Tokenizers for BPE training and PyTorch for AdamW and scaled dot-product attention. [Stanford CS336](https://cs336.stanford.edu/) · [Assignment 1 materials](https://github.com/stanford-cs336/assignment1-basics)
- **CMU 11-711 Fall 2026.** Implement a Llama-style architecture and apply the generation, validation, and SST-5 exercises to your own checkpoint. Use the project tokenizer and FineWeb-Edu throughout instead of the course's TinyStories data. The CMU reference checkpoint uses its own tokenizer. This repo evaluates the pretrained model's generations and SST-5 fine-tuning using the project's vocabulary. [CMU Homework 1](https://cmu-anlp.org/assignments/01)
- **Architecture.** 12 decoder blocks, width 768, 12 query heads, 4 key/value heads, SwiGLU width 3,328, 16,384 vocabulary entries, and context length 256. RMSNorm, RoPE, bias-free projections, and tied input/output embeddings.
