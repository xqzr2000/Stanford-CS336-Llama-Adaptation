"""Codespaces entry points for detached Modal training and evaluation."""

import re
import signal
from contextlib import contextmanager
from pathlib import Path

import modal

ROOT = Path("/artifacts")
DATA = ROOT / "fineweb-edu-10bt"
APP_NAME = "cs336-llama-adaptation"
UV_VERSION = "0.13.6"
MAX_MINUTES = 330
STALL_MINUTES = 15
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(f"{APP_NAME}-artifacts", create_if_missing=True)
locks = modal.Dict.from_name(f"{APP_NAME}-locks", create_if_missing=True)
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extras=["cpu"], extra_options="--no-dev", uv_version=UV_VERSION)
    .add_local_python_source("llama_lab")
)
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extras=["cuda"], extra_options="--no-dev", uv_version=UV_VERSION)
    .add_local_python_source("llama_lab")
)


def safe_id(value):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", value):
        raise ValueError("Use a run ID of 1–64 letters, digits, hyphens or underscores.")
    return value


@contextmanager
def claim(key):
    owner = modal.current_function_call_id()
    if owner is None:
        raise RuntimeError("A Modal function call ID is required.")
    if not locks.put(key, owner, skip_if_exists=True) and locks.get(key) != owner:
        raise RuntimeError(f"{key} already has a writer ({locks.get(key)}).")
    print(f"writer_lock={key} call_id={owner}", flush=True)
    try:
        volume.reload()
        yield
    finally:
        locks.pop(key, None)


@contextmanager
def stall_guard(seconds=STALL_MINUTES * 60):
    """Fail the job if training makes no progress for the given time."""
    def stalled(*_):
        raise RuntimeError(f"No training progress for {seconds} seconds. Stopping to limit spend.")

    try:
        previous = signal.signal(signal.SIGALRM, stalled)
    except ValueError:  # Signals are unavailable outside the main thread.
        print("stall_guard=unavailable", flush=True)
        yield lambda: None
        return
    try:
        yield lambda: signal.alarm(seconds)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@app.function(
    image=cpu_image, volumes={str(ROOT): volume}, cpu=8, memory=16384, timeout=3 * 60 * 60,
)
def prepare(train_shards: int = 3):
    """Download FineWeb-Edu shards and write the tokenizer and packed token streams."""
    from llama_lab.data import prepare_data
    from llama_lab.model import ModelConfig

    with claim("data"):
        manifest = prepare_data(DATA, ModelConfig().vocab_size, train_shards, scratch="/tmp/fineweb")
        volume.commit()
        print(manifest, flush=True)


@app.function(
    image=gpu_image, gpu="H100", volumes={str(ROOT): volume},
    cpu=4, memory=24576, timeout=(MAX_MINUTES + 30) * 60,
)
def pretrain(
    run_id: str = "llama124m", target_step: int = 100, minutes: int = 20,
    schedule_steps: int = 150_000, warmup: int = 200, micro_batch: int = 64,
    accumulation: int = 1, peak_lr: float = 0.0003, checkpoint_every: int = 2_000,
):
    """Create or resume a single pretraining run."""
    import torch

    from llama_lab.data import read_manifest
    from llama_lab.model import ModelConfig
    from llama_lab.train import pretrain_model

    if not 0 < minutes <= MAX_MINUTES:
        raise ValueError(f"Choose 1–{MAX_MINUTES} minutes per invocation. Resume for longer runs.")
    run_id = safe_id(run_id)
    with claim(f"pretrain:{run_id}"), stall_guard() as heartbeat:
        manifest = read_manifest(DATA)
        pretrain_model(
            ModelConfig(vocab_size=manifest["vocab_size"]), DATA, ROOT / "runs" / run_id,
            manifest, torch.device("cuda"), target_step=target_step,
            schedule_steps=schedule_steps, warmup=warmup, micro_batch=micro_batch,
            accumulation=accumulation, peak_lr=peak_lr, minutes=minutes,
            checkpoint_every=checkpoint_every, log_every=25, commit=volume.commit,
            heartbeat=heartbeat,
        )


@app.function(image=cpu_image, volumes={str(ROOT): volume}, cpu=4, memory=6144, timeout=900)
def sample(
    run_id: str = "llama124m", prompt: str = "", max_new: int = 96,
    temperature: float = 0.7, top_p: float = 0.9,
):
    """Read the latest committed checkpoint and save a set of text samples."""
    import json
    import torch

    from llama_lab.data import write_json
    from llama_lab.evaluate import sample_stories

    torch.set_num_threads(4)
    run_root = ROOT / "runs" / safe_id(run_id)
    result = sample_stories(
        run_root / "latest.pt", DATA, torch.device("cpu"), prompt, max_new, temperature, top_p,
    )
    call_id = modal.current_function_call_id()
    path = run_root / f"samples-{result['step']}-{call_id}.json"
    write_json(path, result)
    volume.commit()
    print(json.dumps(result, indent=2), flush=True)
    print(f"saved {path}", flush=True)


@app.function(
    image=gpu_image, gpu="H100", volumes={str(ROOT): volume},
    cpu=4, memory=8192, timeout=90 * 60,
)
def sentiment(
    run_id: str = "llama124m", experiment: str = "sst5", epochs: int = 3,
    batch_size: int = 32, seed: int = 336,
):
    """Fine-tune a copy of the latest checkpoint on SST-5."""
    import torch

    from llama_lab.evaluate import finetune_sst5

    if not 1 <= epochs <= 5 or not 1 <= batch_size <= 64:
        raise ValueError("Choose 1–5 epochs and a batch size of 1–64.")
    run_id, experiment = safe_id(run_id), safe_id(experiment)
    run_root = ROOT / "runs" / run_id
    with claim(f"sst5:{run_id}"):
        finetune_sst5(
            run_root / "latest.pt", DATA, run_root / "sst5-data", run_root / "sentiment" / experiment,
            torch.device("cuda"), epochs, batch_size, seed, volume.commit,
        )
