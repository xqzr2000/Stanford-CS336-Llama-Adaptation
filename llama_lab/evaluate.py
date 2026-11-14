"""Sample continuations and adapt a pretrained checkpoint to five-way SST-5."""

import json
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from llama_lab.data import END, PAD, download, read_manifest, write_json
from llama_lab.train import SEED, atomic_save, precision, restore_model

PROMPTS = (
    "The water cycle describes how water moves between the oceans, the air and the land.",
    "Photosynthesis is the process by which plants",
    "During the Industrial Revolution, many families moved from farms to cities because",
)
SST_REVISION = "e51bdcd8cd3a30da231967c1a249ba59361279a3"


def load_for_eval(checkpoint_path, data_root, device):
    model, metadata = restore_model(checkpoint_path, device)
    if metadata["tokenizer_sha256"] != read_manifest(data_root)["tokenizer_sha256"]:
        raise ValueError("Checkpoint and tokenizer do not match.")
    tokenizer = Tokenizer.from_file(str(data_root / "tokenizer.json"))
    return model, tokenizer, metadata


@torch.inference_mode()
def generate(model, tokenizer, prompt, device, *, max_new, temperature, top_p):
    if max_new < 1 or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("Require max_new > 0, temperature >= 0 and 0 < top_p <= 1.")
    eos = tokenizer.token_to_id(END)
    pad = tokenizer.token_to_id(PAD)
    tokens = tokenizer.encode(prompt).ids or [eos]
    generated = []
    model.eval()
    for _ in range(max_new):
        context = torch.tensor([tokens[-model.config.context:]], device=device)
        with precision(device):
            logits = model(context)[0, -1].float()
        logits[pad] = -torch.inf
        if temperature == 0:
            token = logits.argmax().item()
        else:
            scores, ids = (logits / temperature).sort(descending=True)
            probabilities = scores.softmax(dim=0)
            scores[probabilities.cumsum(0) - probabilities >= top_p] = -torch.inf
            token = ids[torch.multinomial(scores.softmax(dim=0), 1)].item()
        if token == eos:
            break
        tokens.append(token)
        generated.append(token)
    return tokenizer.decode(generated)


def sample_stories(checkpoint, data_root, device, prompt, max_new, temperature, top_p):
    torch.manual_seed(SEED)
    model, tokenizer, metadata = load_for_eval(checkpoint, data_root, device)
    return {
        **metadata, "temperature": temperature, "top_p": top_p, "seed": SEED,
        "samples": [
            {"prompt": text, "continuation": generate(
                model, tokenizer, text, device, max_new=max_new, temperature=temperature, top_p=top_p,
            )}
            for text in ([prompt] if prompt else PROMPTS)
        ],
    }


class SentimentClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(backbone.config.width, 5)

    def forward(self, tokens, lengths):
        features = self.backbone.features(tokens)
        pooled = features[torch.arange(len(tokens), device=tokens.device), lengths - 1]
        return self.head(self.dropout(pooled))


def prepare_sst5(root, tokenizer, context):
    root.mkdir(parents=True, exist_ok=True)
    splits = {}
    eos = tokenizer.token_to_id(END)
    for split in ("train", "dev"):
        filename = f"{split}.jsonl"
        path = root / filename
        download(f"https://huggingface.co/datasets/SetFit/sst5/resolve/{SST_REVISION}/{filename}", path)
        with path.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        encodings = tokenizer.encode_batch([row["text"] for row in records])
        splits[split] = [
            (encoding.ids[:context - 1] + [eos], row["label"])
            for encoding, row in zip(encodings, records, strict=True)
        ]
    return splits


def collate(rows, pad):
    lengths = torch.tensor([len(tokens) for tokens, _ in rows])
    batch = torch.full((len(rows), int(lengths.max()),), pad, dtype=torch.long)
    for index, (tokens, _) in enumerate(rows):
        batch[index, :len(tokens)] = torch.tensor(tokens)
    return batch, lengths, torch.tensor([label for _, label in rows])


@torch.inference_mode()
def accuracy(model, batches, device):
    model.eval()
    correct = total = 0
    for tokens, lengths, labels in batches:
        with precision(device):
            prediction = model(tokens.to(device), lengths.to(device)).argmax(1)
        correct += (prediction == labels.to(device)).sum().item()
        total += len(labels)
    return correct / total


def finetune_sst5(
    checkpoint: Path, data_root, sst_root, output, device, epochs, batch_size, seed, commit,
):
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    torch.manual_seed(seed)
    backbone, tokenizer, metadata = load_for_eval(checkpoint, data_root, device)
    model = SentimentClassifier(backbone).to(device)
    splits = prepare_sst5(sst_root, tokenizer, min(backbone.config.context, 128))
    pad = tokenizer.token_to_id(PAD)
    training = DataLoader(
        splits["train"], batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), collate_fn=lambda rows: collate(rows, pad),
    )
    dev = DataLoader(splits["dev"], batch_size=batch_size, collate_fn=lambda rows: collate(rows, pad))
    majority_label = Counter(label for _, label in splits["train"]).most_common(1)[0][0]
    baseline = sum(label == majority_label for _, label in splits["dev"]) / len(splits["dev"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, fused=device.type == "cuda")
    output.mkdir(parents=True, exist_ok=True)
    best = -1.0
    scores = []
    for epoch in range(1, epochs + 1):
        model.train()
        for tokens, lengths, labels in training:
            optimizer.zero_grad(set_to_none=True)
            with precision(device):
                logits = model(tokens.to(device), lengths.to(device))
                loss = F.cross_entropy(logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        score = accuracy(model, dev, device)
        scores.append(score)
        if score > best:
            best = score
            atomic_save({"model": model.state_dict(), "source": metadata, "epoch": epoch}, output / "best.pt")
        write_json(output / "metrics.json", {
            **metadata, "dataset_revision": SST_REVISION, "seed": seed,
            "majority_dev_accuracy": baseline, "dev_accuracy_by_epoch": scores, "best_dev_accuracy": best,
        })
        commit()
        print(f"sst5 epoch={epoch} dev_accuracy={score:.4f} majority_baseline={baseline:.4f}", flush=True)
