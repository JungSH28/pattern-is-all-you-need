import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import load_wikitext2

torch.manual_seed(0)
DEV = "cpu"
SEQ = 16
EPOCHS = 8
D = int(sys.argv[1]) if len(sys.argv) > 1 else 96


class NeuralBigram(nn.Module):
    """logits = embed(prev) @ Wout. No context, no attention.
    = low-rank factorization of bigram matrix. Isolates embedding generalization."""

    def __init__(self, vocab, d=D):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.out = nn.Linear(d, vocab)

    def forward(self, x):
        return self.out(self.emb(x))


def main():
    tr, va, vocab, _ = load_wikitext2(max_vocab=2000, seq_len=SEQ, batch_size=32)
    V = len(vocab)
    m = NeuralBigram(V).to(DEV)
    n = sum(p.numel() for p in m.parameters())
    print(f"d={D} params={n} ({n/1e6:.2f}M)", flush=True)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    for ep in range(1, EPOCHS + 1):
        m.train()
        for xb, yb in tr:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            loss = F.cross_entropy(m(xb).reshape(-1, V), yb.reshape(-1))
            loss.backward()
            opt.step()
        m.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(DEV), yb.to(DEV)
                l = F.cross_entropy(
                    m(xb).reshape(-1, V), yb.reshape(-1), reduction="sum"
                )
                tot += l.item()
                cnt += yb.numel()
        ppl = math.exp(tot / cnt) if tot / cnt < 700 else float("inf")
        print(f"epoch {ep} : val ppl={ppl:.0f}", flush=True)


if __name__ == "__main__":
    main()
