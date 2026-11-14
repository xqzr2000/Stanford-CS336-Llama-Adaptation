# Cloud walker — Modal

A chronological journal of running this project on Modal. Record what happened to the jobs, what they cost, and how we recovered when something went wrong. The [learning log](Learning-Progress.md) records the model and experiments. The [runbook](Runbook.md) holds the commands.

## November 13, 2026 · Set up the cloud workflow

Codespaces is the control computer. It runs the Modal CLI, starts jobs, and reads logs and artifacts. On Modal, CPU functions prepare FineWeb-Edu and generate samples. One H100 runs pretraining or SST-5 fine-tuning. The named Volume `cs336-llama-adaptation-artifacts` stores the tokenizer, token streams, checkpoints, and results across invocations.

The repository's devcontainer, locked dependencies, and CPU CI are in place. Local verification covered preparation of the full dataset and checkpoint resume. First live preparation, CUDA image build, H100 timings, and actual Modal spend are pending.

## Lessons carried forward from CMU 11-768

| What happened during 11-768 | What we carry into this project |
| --- | --- |
| The workspace showed a $20 spend limit before a planned larger run. We caught it on the billing page and adjusted the schedule while the limit increase was being handled. | Check both limits before a long run. Modal's workspace usage limit applies before credits, and the spend limit applies to out-of-pocket charges. Our **US$200** is a project budget. Read current headroom in the workspace. |
| An app was stopped during the next experiment's cold start. The experiment failed, while the Volume still held the earlier checkpoints. | Check the app ID and wait for a committed checkpoint before assuming work can resume. Record the last committed step if an app stops unexpectedly. |
| The TextCraft trainer continued after the Codespace closed once it ran in Modal. The overnight run later hit stalled requests. | Submit jobs with `modal run --detach`, save each app ID, and use logs and checkpoint metrics to verify progress after reconnecting. An active app and an advancing training step are two different signals. |
| First-step timing was much slower on a cold trainer than on a warmed server. | Record image build, resource wait, initialization, and steady training separately where the logs allow. Forecast the next segment using steady throughput plus the observed overhead. |

[11-768 Modal journal](https://github.com/xqzr2000/cmu-11-768-assignment-3/blob/main/docs/cloud-walker-Modal.md) · [Modal budgets](https://modal.com/docs/guide/budgets) · [Detached runs](https://modal.com/docs/cli/latest/run) · [Volumes](https://modal.com/docs/guide/volumes)

## What to measure on the first H100 pilot

- Record the launch time, app ID, first training log, final checkpoint step, elapsed time, sustained tokens/sec, and peak allocated GPU memory.
- Check that `latest.pt` and the corresponding `metrics-*.json` are visible from Codespaces. On the next segment, record the logged resume step.
- Record billed usage for the app and cumulative project charges when they appear in Modal. Label estimates and dashboard totals separately. Keep pretraining and SST-5 app IDs with the records.
- If the job fails, capture the first relevant error and the last successful checkpoint or log line. Note what changed before the retry and how the retry was verified.

## Journal entry template

Append a dated entry above this template.

- **Workload** — Date, app ID, run ID, stage, CPU/GPU, and intended runtime.
- **Timeline** — Launch, first progress, last checkpoint, and end state.
- **Measurements** — Throughput, peak memory, GPU time, estimated or billed cost, and running project total.
- **Issue and recovery** — Symptom, evidence, diagnosis, fix, and the result of the next attempt.
- **Decision** — What to change for the next session and why.
