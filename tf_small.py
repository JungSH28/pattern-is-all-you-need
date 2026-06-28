import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import load_wikitext2

torch.manual_seed(0)
DEV = "cpu"
SEQ = 16
D = 96
L = 2
H = 4
EPOCHS = 8


class TLM(nn.Module):
    def __init__(self, vocab, d=D, nhead=H, layers=L, seq=SEQ):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        enc = nn.TransformerEncoderLayer(
            d, nhead, dim_feedforward=4 * d, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc, layers)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # tied
        self.seq = seq

    def forward(self, x):
        T = x.size(1)
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        h = self.enc(h, mask=mask)
        return self.head(self.ln(h))


def main():
    tr, va, vocab, _ = load_wikitext2(max_vocab=2000, seq_len=SEQ, batch_size=32)
    V = len(vocab)
    m = TLM(V).to(DEV)
    n = sum(p.numel() for p in m.parameters())
    # tied head shares tok.weight -> counted once already
    print(f"params={n} ({n/1e6:.2f}M)", flush=True)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    for ep in range(1, EPOCHS + 1):
        m.train()
        for xb, yb in tr:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            out = m(xb)
            loss = F.cross_entropy(out.reshape(-1, V), yb.reshape(-1))
            loss.backward()
            opt.step()
        m.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(DEV), yb.to(DEV)
                out = m(xb)
                l = F.cross_entropy(
                    out.reshape(-1, V), yb.reshape(-1), reduction="sum"
                )
                tot += l.item()
                cnt += yb.numel()
        ppl = math.exp(tot / cnt) if tot / cnt < 700 else float("inf")
        print(f"epoch {ep} : val ppl={ppl:.0f}", flush=True)


if __name__ == "__main__":
    main()
