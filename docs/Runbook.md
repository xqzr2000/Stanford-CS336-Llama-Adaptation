# Runbook

How to run Track 3 (the GPT-2 XL-class model on Modal) safely, phase by phase. The [plan](Plan.md) says what each phase does and what it may cost. This page says how to run it without surprises.

The training code doesn't exist yet. Each phase's pull request adds its exact commands to the matching section below. Until then, this page holds the checks that apply to every paid run.

## Before every paid run

1. **Budget.** In the Modal workspace, check **Usage & Billing**: the remaining credits, the workspace budget limit and the spend limit. Set the limit just above the phase's cap in the [plan](Plan.md#phases-gates-and-spending-caps). A limit that stops a run is cheaper than an overrun.
2. **Gate.** Confirm the previous phase's gate was met and recorded in the [learning log](Learning-Progress.md).
3. **Code.** Run the CPU tests on the exact commit you'll launch, and note the commit hash.
4. **Estimate.** Write down the expected run time and cost: tokens ÷ measured tokens/s × USD 35 per node-hour for 8×H100. Before the pilot, use the plan's estimates.
5. **Stop condition.** Decide in advance when you'll stop the run by hand: a time, a step or a cost.

## Launch, watch, stop

Run these from Codespaces with the Modal CLI. They're standard Modal commands; only the app and function names will change per phase.

```bash
m() { uv run --locked modal "$@"; }   # define once per shell
m run --detach <app_file>.py::<function> [--options]   # starts the job; note the app ID it prints
m app list                                             # running apps
m app logs ap-APP_ID                                   # follow the log
m app stop ap-APP_ID                                   # stop a job (it should save a checkpoint first)
m volume ls <volume> <path>                            # check checkpoints and metrics
m volume get <volume> <path> -                         # read a small file, e.g. metrics
```

- **Always `--detach`.** The job keeps running after Codespaces closes. Save each app ID in the [Modal journal](cloud-walker-Modal.md).
- **Progress, not just "running".** An app can be running without training. Check that the logged step and the latest checkpoint advance.
- **Stop cleanly.** The trainer will save a checkpoint at a fixed interval and on a stop signal, so a stopped job resumes from its last checkpoint on the next launch. Phase 3 tests this.

## Phase 1 · Data

To be filled in with the data PR: the tokenizer training and tokenization commands, the expected manifest fields, and how to verify the token counts and checksums.

## Phase 3 · Pilot

To be filled in with the trainer PR. What to record:

- Image build time, time waiting for GPUs, start-up time and time to the first step, each separately.
- Tokens/s and MFU once steady, peak GPU memory per device, and the time to save a checkpoint.
- A stop and resume across two function calls, with the step and loss on both sides.
- The billed cost of the pilot, from Modal's usage page.

## Phase 5 · Full run

To be filled in after the pilot: launch command, checkpoint interval, the cost-based stop rule, and the milestone checkpoints to keep.

## Recover from a failed or interrupted job

| Symptom | What to do |
| --- | --- |
| The job stopped or timed out | Check the last committed checkpoint and its step, then relaunch with the same run ID. It resumes there. |
| The app is running but the step doesn't advance | Stop it. Look for the first error in the log, record it in the journal, and fix it before relaunching. The stall guard should catch this on its own. |
| Loss spike or NaN | The spike guard (as in Track 2's notebook) rolls back automatically. If it keeps firing, stop and lower the learning rate or check the data at those steps. |
| Out of memory | Lower the per-GPU micro-batch. Gradient accumulation keeps the global batch, and the run's math, unchanged. |
| The budget limit stopped the job | Don't raise the limit mid-phase without updating the plan's cap and the reason in the journal. |
