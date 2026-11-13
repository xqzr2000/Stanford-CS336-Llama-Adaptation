"""Prepare pinned TinyStories splits using a new byte-level BPE tokenizer."""

import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from itertools import batched, islice
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

END = "<|endoftext|>"
PAD = "<|pad|>"
DATASET = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
BPE_TRAIN_STORIES = 100_000


def download(url, path):
    if path.exists():
        return
    partial = path.with_suffix(path.suffix + ".part")
    with urlopen(url, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    partial.replace(path)


def stories(path: Path) -> Iterable[str]:
    with path.open(encoding="utf-8") as source:
        lines = []
        for line in source:
            if line.strip() == END:
                if lines:
                    yield "".join(lines).strip()
                lines.clear()
            else:
                lines.append(line)
        if lines and (tail := "".join(lines).strip()):
            yield tail


def train_tokenizer(texts, vocab_size):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[END, PAD],
        show_progress=False,
    )
    tokenizer.train_from_iterator(batched(texts, 512), trainer)
    return tokenizer


def encode_split(path, tokenizer, output):
    eos = tokenizer.token_to_id(END)
    count = 0
    with output.open("wb") as stream:
        for batch in batched(stories(path), 512):
            ids = []
            for encoding in tokenizer.encode_batch(list(batch)):
                ids.extend(encoding.ids)
                ids.append(eos)
            np.asarray(ids, dtype="<u2").tofile(stream)
            count += len(ids)
    return count


def write_json(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def prepare_data(root, vocab_size):
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = read_manifest(root)
        if manifest["vocab_size"] != vocab_size or manifest["revision"] != REVISION:
            raise ValueError("Existing dataset uses a different tokenizer size or revision.")
        return manifest

    splits = {}
    for split in ("train", "valid"):
        filename = f"TinyStoriesV2-GPT4-{split}.txt"
        splits[split] = root / filename
        download(f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{filename}", splits[split])

    tokenizer = train_tokenizer(islice(stories(splits["train"]), BPE_TRAIN_STORIES), vocab_size)
    if tokenizer.get_vocab_size() != vocab_size:
        raise ValueError(f"BPE produced {tokenizer.get_vocab_size()} tokens. Expected {vocab_size}.")
    tokenizer_path = root / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    counts = {}
    for split, source in splits.items():
        partial = root / f"{split}.bin.part"
        counts[split] = encode_split(source, tokenizer, partial)
        partial.replace(root / f"{split}.bin")
        print(f"prepared split={split} tokens={counts[split]:,}", flush=True)
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "vocab_size": vocab_size,
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "tokens": counts,
    }
    write_json(manifest_path, manifest)
    return manifest


def read_manifest(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    fingerprint = hashlib.sha256((root / "tokenizer.json").read_bytes()).hexdigest()
    if manifest["tokenizer_sha256"] != fingerprint:
        raise ValueError("Tokenizer hash does not match the data manifest.")
    for split, count in manifest["tokens"].items():
        if (root / f"{split}.bin").stat().st_size != count * 2:
            raise ValueError(f"{split}.bin has the wrong size. Prepare the dataset again.")
    return manifest
