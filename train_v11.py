"""
0단계 v11 — 최소 모델. 순수 Hebbian 공동출현 + log(Weber-Fechner) readout.
학습=한 번 카운트(에폭 불필요). 목표 ppl ~352 (v9 936 돌파), 천장 bigram 78.
"""
import math
import torch
import torch.nn.functional as F

from data import load_wikitext2
from model_v11 import SparsePatternLMv11

VOCAB, N = 2000, 4000
BS, SEQ = 32, 16
PAD = 0


def bigram_counts(train_batches, V):
    C = torch.zeros(V, V)
    for inp, _ in train_batches:
        for seq in inp.tolist():
            for i in range(len(seq) - 1):
                a, b = seq[i], seq[i + 1]
                if a != PAD and b != PAD:
                    C[a, b] += 1
    return C


@torch.no_grad()
def evaluate(model, val_batches, device, log=True):
    tot, n = 0.0, 0
    for inp, _ in val_batches:
        inp = inp.to(device)
        for seq in inp:
            s = seq.tolist()
            for i in range(len(s) - 1):
                if s[i] == PAD or s[i + 1] == PAD:
                    continue
                prev = seq[i:i + 1]
                lg = model.logits(prev) if log else (model.anchor[prev] @ model.W) @ model.anchor.t()
                tot += F.cross_entropy(lg, seq[i + 1:i + 2], ignore_index=PAD).item()
                n += 1
    return tot / n


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    train_batches, val_batches, vocab, _ = load_wikitext2(max_vocab=VOCAB, seq_len=SEQ, batch_size=BS)
    print(f"train={len(train_batches)} val={len(val_batches)}")

    C = bigram_counts(train_batches, VOCAB)
    model = SparsePatternLMv11(vocab_size=VOCAB, N=N, k0=20).to(device)
    model.fit(C)

    vl_log = evaluate(model, val_batches, device, log=True)
    nz = (model.W.abs() > 1e-6).float().mean().item()
    pl = math.exp(vl_log) if vl_log < 700 else float("inf")
    print(f"v11 (Hebbian + log readout): val={vl_log:.3f} ppl={pl:.0f} | W_nz={nz:.3f}")
    print("(목표 ~352 / unigram 125 / bigram 78 / v9 936)")
    torch.save({"W": model.W, "anchor": model.anchor, "k0": 20}, "model_v11_best.pt")


if __name__ == "__main__":
    main()
