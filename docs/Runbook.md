# Runbook

Run these commands from the repository root in Codespaces. Replace app ID examples with the actual `ap-…` IDs. Use the same Modal workspace and environment throughout.

## Day 1 · Set up and prepare FineWeb-Edu

```bash
git pull --ff-only
uv sync --locked --extra cpu
uv run --locked --extra cpu python -m pytest -q
m() { uv run --locked --extra cpu modal "$@"; }
```

Define `m` again in each new shell. If the Codespace needs credentials, run `m token new` and select the intended workspace. You can also grant the repo access to existing Codespaces secrets named `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. Check **Usage & Billing** for the workspace's remaining usage and spend limits.

```bash
m app list
m run --detach modal_app.py::prepare
```

Save the app ID. Wait for Modal to start the app before closing Codespaces. On the next check-in, confirm that preparation completed and that the manifest is readable.

```bash
m app logs ap-PREP_ID
m volume get cs336-llama-adaptation-artifacts fineweb-edu-10bt/manifest.json -
```

The manifest records the dataset revision, shard list, vocabulary size, tokenizer hash, and token counts. Rerunning preparation after completion reuses the prepared files. The logs show download and encoding time for each shard; record them in the Modal journal. Copy the training token count into the [cost estimate](Plan.md#cost-estimate) in place of the 2.5B estimate.

## Day 2 · Pilot the full-size model

```bash
m run --detach modal_app.py::pretrain --run-id pilot --target-step 100 --minutes 20
m app logs ap-PILOT_ID
m volume ls cs336-llama-adaptation-artifacts runs/pilot
```

Confirm the reported parameter count of **123,489,024**, the initial validation loss, progress logs with finite loss, and a committed `latest.pt` with a matching `metrics-*.json` file. At step 100 the metrics file is `metrics-000100.json`. If the time limit stopped training earlier, use the actual saved step.

```bash
m volume get cs336-llama-adaptation-artifacts runs/pilot/metrics-000100.json -
```

The metrics file also holds `train_tokens_per_second`, the latest training loss, and `validation_and_save_seconds`. Read the logs on the same day if you need them. The Starter plan keeps logs for one day.

## Days 3–5 · Start the main run, budget, and sample

Put the pilot's `train_tokens_per_second` into the [cost estimate](Plan.md#cost-estimate) and choose the schedule length. The default is 150,000 steps, about one pass over the three default training shards. For a different length, add the same `--schedule-steps N` to every pretraining command below. The checkpoint rejects a segment with a different schedule, warmup, batch setting, or peak learning rate.

Start the main run under the run ID `llama124m`. Once a pretraining app has finished, submit the next segment under the same run ID. `--target-step` is cumulative.

```bash
m run --detach modal_app.py::pretrain --run-id llama124m --target-step 1000 --minutes 90
m app logs ap-RESUME_ID
```

On segments after the first, look for `resumed step=...` with the prior saved step. `--minutes` accepts at most 330. Calculate remaining tokens as `(target step − saved step) × 16,384` with the default batch. Divide by measured tokens per second and then by 3,600 for estimated GPU hours. Use several steady log windows, then allow for initialization, checkpointing, evaluation, and CPU/RAM. Check actual charges before scheduling more work.

```bash
m run modal_app.py::sample --run-id llama124m --max-new 96 --temperature 0.7 --top-p 0.9
```

Use the same prompts and settings after later segments. Sampling prints the text and stores a JSON record in `runs/llama124m`. Keep the checkpoint step with any excerpts copied into the learning log.

## Day 6 · Export to Colab and fine-tune on SST-5

```bash
m run modal_app.py::export --run-id llama124m
m volume get cs336-llama-adaptation-artifacts runs/llama124m/colab llama124m-colab
```

The export reads the latest committed checkpoint and replaces any earlier `colab` folder for the run. Upload `llama124m-colab` to Google Drive, [open the notebook in Colab](https://colab.research.google.com/github/xqzr2000/Stanford-CS336-Llama-Adaptation/blob/main/notebooks/CS336_Llama_Project_Walkthrough.ipynb) with a T4 runtime, set `MODEL_FOLDER` and `MOUNT_DRIVE`, and run chapters 4 and 5 with `SST5_MODEL = "final"`. Results are saved to `sst5-colab-seed<seed>` in the Drive folder.

If Colab is unavailable, the same fine-tuning runs on Modal:

```bash
m run --detach modal_app.py::sentiment --run-id llama124m --experiment sst5
m app logs ap-SST5_ID
m volume get cs336-llama-adaptation-artifacts runs/llama124m/sentiment/sst5/metrics.json -
```

Either way, fine-tuning starts from the latest committed pretraining checkpoint. Record the source step from the metrics, the seed, accuracy by epoch, and the majority-class baseline. The best classifier checkpoint is saved under the experiment directory. If you repeat with seed 337, choose a fresh experiment name and keep the source checkpoint fixed for the comparison.

```bash
m run --detach modal_app.py::sentiment --run-id llama124m --experiment sst5-seed337 --seed 337
```

## Day 7 · Close out

Read the latest pretraining and sentiment metrics, check which apps are still running, and reconcile their charges in Modal. Add the model results to [Learning-Progress.md](Learning-Progress.md) and the infrastructure and cost results to [cloud-walker-Modal.md](cloud-walker-Modal.md).

## Recover from a failed or interrupted job

| Symptom | What to check or do |
| --- | --- |
| Missing manifest or checkpoint | Check the app state and the selected workspace/environment. Read the prep or training logs and confirm the files were committed to the expected Volume before running the next stage. |
| Startup looks stalled | Check whether Modal is building the image, waiting for resources, or initializing the job. Use the app ID to inspect logs before submitting a duplicate. |
| Job failed with `No training progress` | The stall guard ended a job that completed no step or checkpoint for 15 minutes. Read the last log lines to see where it stopped, confirm the last committed metrics, then resubmit the same command. |
| Preparation fails with HTTP 404 | The pinned FineWeb-Edu revision or shard name no longer resolves. Check `REVISION` and the shard names in `llama_lab/data.py` against the dataset's file list on Hugging Face. |
| Preparation fails with HTTP 429 | Hugging Face rate-limited the anonymous download after four attempts. Wait a few minutes and rerun, or attach a Modal Secret holding `HF_TOKEN` to `prepare`. |
| Run stopped before its target | Read the last committed metrics. When ready, submit the same run ID and cumulative target to resume. An interruption before the first checkpoint means the run starts again from random weights. |
| Checkpoint configuration mismatch | Restore the original schedule, batch, and learning-rate settings. Use a new run ID if you intend to change the experiment. |
| Writer lock remains after an interruption | Inspect it with `m dict items cs336-llama-adaptation-locks`. If **all** preparation, pretraining, and sentiment apps for this project have stopped and the lock is stale, clear it with `m dict clear cs336-llama-adaptation-locks`. |
| Billing limit stops work | Check the workspace's usage and spend limits and the project's remaining budget. Confirm the latest committed checkpoint before resuming. |

To stop a specific app, use `m app stop ap-APP_ID` or the Modal dashboard. Record the last committed step and any work lost after it. Strip credentials and environment variable values from copied logs.

[Modal CLI](https://modal.com/docs/cli/latest/run) · [Volumes](https://modal.com/docs/guide/volumes) · [Workspace budgets](https://modal.com/docs/guide/budgets)
