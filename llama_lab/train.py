"""Checkpointable next-token training on a single device."""

import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from llama_lab.data import write_json
from llama_lab.model import Llama, ModelConfig

SEED = 336


def atomic_save(payload, path):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=True)


def windows(tokens, starts, context, device):
    indices = starts.numpy()[:, None] + np.arange(context + 1)
    packed = torch.from_numpy(tokens[indices].astype(np.int64)).to(device)
    return packed[:, :-1].contiguous(), packed[:, 1:].contiguous()


def precision(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def learning_rate(step, schedule_steps, warmup, peak):
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / (schedule_steps - warmup)
    return peak * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)


def lm_loss(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits.flatten(0, 1), y.flatten())


@torch.inference_mode()
def validate(model, tokens, micro_batch, device, batches=8):
    context = model.config.context
    count = min(micro_batch * batches, (len(tokens) - 1) // context)
    if count < 1:
        raise ValueError("Validation split is too short for a full context.")
    starts = torch.linspace(0, len(tokens) - context - 1, count, dtype=torch.float64).long()
    model.eval()
    loss = 0.0
    for chunk in starts.split(micro_batch):
        x, y = windows(tokens, chunk, context, device)
        with precision(device):
            loss += lm_loss(model, x, y).item() * len(chunk)
    model.train()
    return loss / count


def pretrain_model(
    config, data_root, run_root, fingerprint, device, *,
    target_step, schedule_steps, warmup, micro_batch, accumulation, peak_lr,
    minutes, checkpoint_every, log_every, commit=lambda: None,
):
    if not (0 < target_step <= schedule_steps and 0 < warmup < schedule_steps):
        raise ValueError("Require 0 < target_step <= schedule_steps and 0 < warmup < schedule_steps.")
    if min(micro_batch, accumulation, minutes, checkpoint_every, log_every) <= 0 or peak_lr <= 0:
        raise ValueError("Training sizes, runtime, learning rate and intervals must be positive.")

    started = time.monotonic()
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_root / "latest.pt"
    settings = {
        "model": asdict(config), "data": fingerprint, "schedule_steps": schedule_steps,
        "warmup": warmup, "micro_batch": micro_batch, "accumulation": accumulation,
        "peak_lr": peak_lr, "seed": SEED,
    }
    train_tokens = np.memmap(data_root / "train.bin", dtype="<u2", mode="r")
    valid_tokens = np.memmap(data_root / "valid.bin", dtype="<u2", mode="r")
    if len(train_tokens) <= config.context:
        raise ValueError("Training split is too short for a full context.")

    torch.manual_seed(SEED)
    model = Llama(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1,
        fused=device.type == "cuda",
    )
    batch_rng = torch.Generator().manual_seed(SEED)
    step = 0
    if checkpoint_path.exists():
        state = load_checkpoint(checkpoint_path, "cpu")
        if state["settings"] != settings:
            raise ValueError("Checkpoint configuration differs. Use the original settings or a new run ID.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        batch_rng.set_state(state["batch_rng"])
        step = state["step"]
        del state
        print(f"resumed step={step}", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    tokens_per_step = micro_batch * accumulation * config.context
    print(f"params={parameter_count:,} tokens_per_step={tokens_per_step:,} target_step={target_step}", flush=True)
    if step >= target_step:
        print(f"already reached target step={step}", flush=True)
        return
    if step == 0:
        print(f"step=0 val_loss={validate(model, valid_tokens, micro_batch, device):.4f}", flush=True)

    window_start = time.monotonic()
    window_steps = 0
    window_loss = 0.0
    while step < target_step:
        lr = learning_rate(step + 1, schedule_steps, warmup, peak_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            starts = torch.randint(
                len(train_tokens) - config.context, (micro_batch,), generator=batch_rng,
            )
            x, y = windows(train_tokens, starts, config.context, device)
            with precision(device):
                loss = lm_loss(model, x, y) / accumulation
            loss.backward()
            window_loss += loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        window_steps += 1
        out_of_time = time.monotonic() - started >= minutes * 60
        stopping = step == target_step or out_of_time
        if step % log_every == 0 or stopping:
            mean_loss = window_loss.item() / window_steps
            if not math.isfinite(mean_loss):
                raise RuntimeError(f"Non-finite training loss at step {step}. The prior checkpoint is intact.")
            throughput = window_steps * tokens_per_step / (time.monotonic() - window_start)
            print(
                f"step={step} tokens={step * tokens_per_step:,} train_loss={mean_loss:.4f} "
                f"lr={lr:.3g} train_tokens_per_second={throughput:,.0f}",
                flush=True,
            )
            window_steps, window_loss = 0, 0.0
            window_start = time.monotonic()
        if step % checkpoint_every == 0 or stopping:
            val_loss = validate(model, valid_tokens, micro_batch, device)
            if not math.isfinite(val_loss):
                raise RuntimeError(f"Non-finite validation loss at step {step}. The prior checkpoint is intact.")
            atomic_save({
                "settings": settings, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "batch_rng": batch_rng.get_state(), "step": step, "tokens": step * tokens_per_step,
                "val_loss": val_loss,
            }, checkpoint_path)
            metrics = {
                "step": step, "tokens": step * tokens_per_step, "val_loss": val_loss,
                "perplexity": math.exp(val_loss), "parameters": parameter_count,
                "elapsed_seconds": time.monotonic() - started,
            }
            if device.type == "cuda":
                metrics["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
            write_json(run_root / f"metrics-{step:06d}.json", metrics)
            commit()
            print(f"checkpoint {metrics}", flush=True)
        if out_of_time:
            print(f"runtime limit reached at step={step}", flush=True)
            break


def restore_model(path: Path, device):
    checkpoint = load_checkpoint(path, "cpu")
    model = Llama(ModelConfig(**checkpoint["settings"]["model"]))
    model.load_state_dict(checkpoint["model"])
    metadata = {k: checkpoint[k] for k in ("step", "tokens", "val_loss")}
    metadata["tokenizer_sha256"] = checkpoint["settings"]["data"]["tokenizer_sha256"]
    return model.to(device), metadata
