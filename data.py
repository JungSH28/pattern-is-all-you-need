from collections import Counter

import torch
from datasets import load_dataset


def load_wikitext2(
    max_vocab: int = 2000,
    seq_len: int = 32,
    batch_size: int = 32,
):
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", trust_remote_code=True)

    # Vocab from train split
    counter: Counter = Counter()
    for ex in raw["train"]:
        counter.update(ex["text"].lower().split())

    special = ["<pad>", "<unk>"]
    vocab = special + [w for w, _ in counter.most_common(max_vocab - len(special))]
    w2i = {w: i for i, w in enumerate(vocab)}

    def encode_split(split_name: str) -> list[int]:
        ids: list[int] = []
        for ex in raw[split_name]:
            ids += [w2i.get(t, 1) for t in ex["text"].lower().split()]
        return ids

    def make_batches(ids: list[int]):
        chunks = [
            ids[i : i + seq_len + 1]
            for i in range(0, len(ids) - seq_len - 1, seq_len)
        ]
        batches = []
        for i in range(0, len(chunks) - batch_size, batch_size):
            b = torch.tensor(chunks[i : i + batch_size], dtype=torch.long)
            batches.append((b[:, :-1], b[:, 1:]))  # (input, target)
        return batches

    train_batches = make_batches(encode_split("train"))
    val_batches = make_batches(encode_split("validation"))

    return train_batches, val_batches, vocab, w2i
