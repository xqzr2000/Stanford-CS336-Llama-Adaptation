import json
import shutil
import time

import numpy as np
import pytest
import torch

from llama_lab import data as data_module
from llama_lab.data import END, prepare_data, read_manifest
from llama_lab.evaluate import SentimentClassifier, finetune_sst5, load_for_eval, sample_stories
from llama_lab.model import Llama, ModelConfig
from llama_lab.train import export_weights, load_checkpoint, pretrain_model

CPU = torch.device("cpu")
DOCUMENTS = (
    "Maya found a café beside the blue lake.",
    "Plants turn sunlight, water and carbon dioxide into sugar.",
    "The river carried sand and small stones toward the sea.",
)


@pytest.fixture(autouse=True)
def use_one_thread():
    torch.set_num_threads(1)


def write_shard(path, texts):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"text": list(texts), "id": [str(i) for i in range(len(texts))]}), path)


@pytest.fixture
def tiny_data(tmp_path, monkeypatch):
    source = tmp_path / "hub"
    source.mkdir()
    for name in (*data_module.TRAIN_SHARDS[:2], data_module.VALID_SHARD):
        write_shard(source / name, DOCUMENTS * 16)
    monkeypatch.setattr(data_module, "VALID_TOKENS", 100)
    fetched = []

    def fetch(name, path):
        fetched.append(name)
        shutil.copy(source / name, path)

    root = tmp_path / "data"
    manifest = prepare_data(root, 300, 2, scratch=tmp_path / "scratch", fetch=fetch)
    assert fetched == ["000_00000.parquet", "001_00000.parquet", "013_00000.parquet"]
    assert not list((tmp_path / "scratch").iterdir())
    assert manifest["tokens"]["valid"] == 100
    tokenizer = data_module.Tokenizer.from_file(str(root / "tokenizer.json"))
    assert tokenizer.decode(tokenizer.encode(DOCUMENTS[0]).ids) == DOCUMENTS[0]
    train = np.fromfile(root / "train.bin", dtype="<u2")
    assert train[-1] == tokenizer.token_to_id(END)
    assert (train == tokenizer.token_to_id(END)).sum() == 2 * len(DOCUMENTS) * 16
    assert prepare_data(root, 300, 2, fetch=None) == manifest
    with pytest.raises(ValueError, match="different tokenizer size, revision or shard list"):
        prepare_data(root, 300, 3)
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(), width=16, layers=1,
        query_heads=2, kv_heads=1, hidden_width=32, context=8,
    )
    return config, root, read_manifest(root)


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


def test_weights_export_loads_with_only_the_tokenizer(tiny_data, tmp_path):
    config, data, manifest = tiny_data
    run = tmp_path / "run"
    settings = dict(
        schedule_steps=2, warmup=1, micro_batch=2, accumulation=1,
        peak_lr=1e-3, minutes=10, checkpoint_every=1, log_every=1,
    )
    pretrain_model(config, data, run, manifest, CPU, target_step=1, **settings)
    exported = export_weights(run / "latest.pt", tmp_path / "weights.pt")
    payload = load_checkpoint(exported, CPU)
    assert "optimizer" not in payload and "batch_rng" not in payload
    assert all(t.dtype == torch.float16 for t in payload["model"].values())
    assert exported.stat().st_size < (run / "latest.pt").stat().st_size / 4

    colab = tmp_path / "colab"
    colab.mkdir()
    shutil.copy(data / "tokenizer.json", colab / "tokenizer.json")
    full, _, _ = load_for_eval(run / "latest.pt", colab, CPU)
    light, _, metadata = load_for_eval(exported, colab, CPU)
    assert metadata["step"] == 1
    assert next(light.parameters()).dtype == torch.float32
    tokens = torch.tensor([[1, 2, 3, 4]])
    torch.testing.assert_close(light(tokens), full(tokens), atol=1e-3, rtol=1e-3)

    (colab / "tokenizer.json").write_text((colab / "tokenizer.json").read_text().replace(END, "<|x|>"))
    with pytest.raises(ValueError, match="do not match"):
        load_for_eval(exported, colab, CPU)
