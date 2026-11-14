"""Prepare pinned FineWeb-Edu token streams using a new byte-level BPE tokenizer."""

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterable, Iterator
from itertools import batched, islice
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

END = "<|endoftext|>"
PAD = "<|pad|>"
DATASET = "HuggingFaceFW/fineweb-edu"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SUBSET = "sample/10BT"
# The 10BT sample has 14 shards: 000–012 are about 2.15 GB each and 013 is about 0.54 GB.
TRAIN_SHARDS = tuple(f"{index:03d}_00000.parquet" for index in range(13))
VALID_SHARD = "013_00000.parquet"
DEFAULT_TRAIN_SHARDS = 3
MAX_TRAIN_SHARDS = 6
VALID_TOKENS = 5_000_000
BPE_TRAIN_DOCUMENTS = 100_000


def shard_url(filename):
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{SUBSET}/{filename}"


def download(url, path, attempts=4):
    """Download to a temporary name, retrying rate limits and dropped connections."""
    if path.exists():
        return
    partial = path.with_suffix(path.suffix + ".part")
    headers = {"User-Agent": "cs336-llama-adaptation"}
    if token := os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(Request(url, headers=headers), timeout=120) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=16 << 20)
            partial.replace(path)
            return
        except (HTTPError, URLError, TimeoutError, ConnectionError) as error:
            retryable = not isinstance(error, HTTPError) or error.code in (429, 500, 502, 503, 504)
            if not retryable or attempt == attempts:
                raise
            wait = 30 * attempt
            print(f"download_retry url={url} attempt={attempt} error={error} wait={wait}s", flush=True)
            time.sleep(wait)


def parquet_texts(path: Path) -> Iterator[str]:
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["text"]):
        yield from batch.column(0).to_pylist()


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


def encode_texts(texts: Iterable[str], tokenizer, stream, limit=None):
    """Append documents to an open binary stream, each followed by an end-of-text token."""
    eos = tokenizer.token_to_id(END)
    count = 0
    for batch in batched(texts, 1024):
        ids = []
        for encoding in tokenizer.encode_batch(list(batch)):
            ids.extend(encoding.ids)
            ids.append(eos)
        if limit is not None:
            ids = ids[:limit - count]
        np.asarray(ids, dtype="<u2").tofile(stream)
        count += len(ids)
        if limit is not None and count >= limit:
            break
    return count


def write_json(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def prepare_data(root, vocab_size, train_shards=DEFAULT_TRAIN_SHARDS, *, scratch=None, fetch=None):
    """Write tokenizer.json, train.bin, valid.bin and manifest.json under root.

    Raw shards go to local scratch space one at a time and are deleted after encoding,
    so the Volume holds only the token streams.
    """
    if vocab_size > 65_536:
        raise ValueError("Token IDs are stored as uint16. Use a vocabulary of at most 65,536.")
    if not 1 <= train_shards <= MAX_TRAIN_SHARDS:
        raise ValueError(f"Choose 1–{MAX_TRAIN_SHARDS} training shards.")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    files = {"train": list(TRAIN_SHARDS[:train_shards]), "valid": [VALID_SHARD]}
    if manifest_path.exists():
        manifest = read_manifest(root)
        expected = (vocab_size, REVISION, files)
        if (manifest["vocab_size"], manifest["revision"], manifest["files"]) != expected:
            raise ValueError(
                "The prepared dataset uses a different tokenizer size, revision or shard list. "
                "Remove the dataset folder from the Volume or use the original settings."
            )
        return manifest

    fetch = fetch or (lambda name, path: download(shard_url(name), path))
    scratch = Path(scratch or tempfile.mkdtemp(prefix="fineweb-"))
    scratch.mkdir(parents=True, exist_ok=True)

    def local(name):
        path = scratch / name
        started = time.monotonic()
        fetch(name, path)
        print(f"downloaded {name} bytes={path.stat().st_size:,} seconds={time.monotonic() - started:.0f}", flush=True)
        return path

    first = local(files["train"][0])
    tokenizer = train_tokenizer(islice(parquet_texts(first), BPE_TRAIN_DOCUMENTS), vocab_size)
    if tokenizer.get_vocab_size() != vocab_size:
        raise ValueError(f"BPE produced {tokenizer.get_vocab_size()} tokens. Expected {vocab_size}.")
    tokenizer_path = root / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"trained tokenizer vocab_size={vocab_size} documents={BPE_TRAIN_DOCUMENTS:,}", flush=True)

    counts = {}
    for split, names in files.items():
        partial = root / f"{split}.bin.part"
        counts[split] = 0
        with partial.open("wb") as stream:
            for name in names:
                path = first if path_matches(first, name) else local(name)
                started = time.monotonic()
                limit = VALID_TOKENS if split == "valid" else None
                added = encode_texts(parquet_texts(path), tokenizer, stream, limit)
                counts[split] += added
                path.unlink()
                print(
                    f"encoded split={split} shard={name} tokens={added:,} "
                    f"seconds={time.monotonic() - started:.0f}",
                    flush=True,
                )
        partial.replace(root / f"{split}.bin")
        print(f"prepared split={split} tokens={counts[split]:,}", flush=True)
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "subset": SUBSET,
        "files": files,
        "vocab_size": vocab_size,
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "tokens": counts,
    }
    write_json(manifest_path, manifest)
    return manifest


def path_matches(path, name):
    return path.name == name and path.exists()


def read_manifest(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    fingerprint = hashlib.sha256((root / "tokenizer.json").read_bytes()).hexdigest()
    if manifest["tokenizer_sha256"] != fingerprint:
        raise ValueError("Tokenizer hash does not match the data manifest.")
    for split, count in manifest["tokens"].items():
        if (root / f"{split}.bin").stat().st_size != count * 2:
            raise ValueError(f"{split}.bin has the wrong size. Prepare the dataset again.")
    return manifest
