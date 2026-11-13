import hashlib
import json
import time

import numpy as np
import pytest
import torch

from llama_lab.data import END, encode_split, read_manifest, train_tokenizer, write_json
from llama_lab.evaluate import SentimentClassifier, finetune_sst5, load_for_eval, sample_stories
from llama_lab.model import Llama, ModelConfig
from llama_lab.train import load_checkpoint, pretrain_model

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def use_one_thread():
    torch.set_num_threads(1)


@pytest.fixture
def tiny_data(tmp_path):
    source = tmp_path / "stories.txt"
    line = "Maya found a café beside the blue lake."
    source.write_text(f"{line}\n{END}\nThe bear smiled.\n{END}\n" * 16, encoding="utf-8")
    tokenizer = train_tokenizer([line, "The bear smiled."], 300)
    assert tokenizer.decode(tokenizer.encode(line).ids) == line
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    counts = {
        split: encode_split(source, tokenizer, tmp_path / f"{split}.bin")
        for split in ("train", "valid")
    }
    assert np.fromfile(tmp_path / "train.bin", dtype="<u2")[-1] == tokenizer.token_to_id(END)
    write_json(tmp_path / "manifest.json", {
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(), "tokens": counts,
    })
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(), width=16, layers=1,
        query_heads=2, kv_heads=1, hidden_width=32, context=8,
    )
    return config, tmp_path, read_manifest(tmp_path)


def test_production_parameter_count():
    with torch.device("meta"):
        model = Llama(ModelConfig())
    assert sum(parameter.numel() for parameter in model.parameters()) == 123_489_024


def test_causal_attention_and_padding(tiny_data):
    config, _, _ = tiny_data
    model = Llama(config).eval()
    short = torch.tensor([[2, 3, 4]])
    long = torch.tensor([[2, 3, 4, 5, 6, 7]])
    torch.testing.assert_close(model(short), model(long)[:, :3])
    classifier = SentimentClassifier(model).eval()
    torch.testing.assert_close(classifier(short, torch.tensor([3])), classifier(long, torch.tensor([3])))


def test_resume_generation_and_sentiment(tiny_data):
    config, data, manifest = tiny_data
    run = data / "resume"
    settings = dict(
        schedule_steps=6, warmup=1, micro_batch=2, accumulation=1, peak_lr=0.001,
        minutes=1, checkpoint_every=2, log_every=1,
    )
    beats = []
    pretrain_model(config, data, run, manifest, CPU, target_step=2, heartbeat=lambda: beats.append(1), **settings)
    assert len(beats) >= 4
    metrics = json.loads((run / "metrics-000002.json").read_text())
    assert metrics["train_loss"] > 0 and metrics["train_tokens_per_second"] > 0
    assert metrics["validation_and_save_seconds"] >= 0
    pretrain_model(config, data, run, manifest, CPU, target_step=4, **settings)
    pretrain_model(config, data, data / "straight", manifest, CPU, target_step=4, **settings)
    checkpoint = run / "latest.pt"
    resumed = load_checkpoint(checkpoint, CPU)
    straight = load_checkpoint(data / "straight/latest.pt", CPU)
    assert resumed["step"] == 4
    assert resumed["tokens"] == 64
    assert resumed["val_loss"] == straight["val_loss"]
    for name, tensor in straight["model"].items():
        torch.testing.assert_close(resumed["model"][name], tensor, rtol=0, atol=0)
    with pytest.raises(ValueError, match="Checkpoint configuration differs"):
        pretrain_model(config, data, run, manifest, CPU, target_step=5, **(settings | {"peak_lr": 0.1}))

    result = sample_stories(checkpoint, data, CPU, "Once", 4, 0.7, 0.9)
    assert result["samples"][0]["prompt"] == "Once"

    sst = data / "sst"
    sst.mkdir()
    for split in ("train", "dev"):
        rows = [{"text": "Good.", "label": 4}, {"text": "Bad.", "label": 0}, {"text": "Great.", "label": 4}]
        (sst / f"{split}.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    output = data / "sentiment"
    finetune_sst5(checkpoint, data, sst, output, CPU, 1, 2, 336, lambda: None)
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["majority_dev_accuracy"] == pytest.approx(2 / 3)
    assert 0 <= metrics["best_dev_accuracy"] <= 1
    assert (output / "best.pt").exists()
    _, _, metadata = load_for_eval(checkpoint, data, CPU)
    assert metadata["step"] == 4


def test_stall_guard_stops_a_hung_job():
    from modal_app import stall_guard

    with pytest.raises(RuntimeError, match="No training progress"):
        with stall_guard(seconds=1) as heartbeat:
            heartbeat()
            time.sleep(5)
    with stall_guard(seconds=1) as heartbeat:
        heartbeat()
    time.sleep(1.5)
