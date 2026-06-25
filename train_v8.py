"""
0단계 v8 — 도파민 R-STDP. 예측오차 δ 게이팅 heteroassociative 학습.
문맥상태 h_t → 다음토큰 anchor로, δ=1−p(correct) 게이팅.
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import load_wikitext2
from model_v8 import SparsePatternLMv8

VOCAB, N = 2000, 4000
EPOCHS, BS, SEQ = 4, 32, 16
CFG = dict(rho=0.06, gamma=4, theta_init=0.3, k0=20, eta_homeo=0.02)


@torch.no_grad()
def evaluate(model, val_batches, device):
    total, n = 0.0, 0
    for inp, _ in val_batches:
        inp = inp.to(device)
        B, T = inp.shape
        h = model.new_state(B, device)
        loss = 0.0
        for t in range(T - 1):
            h = model.integrate(h, inp[:, t])
            loss += F.cross_entropy(model.drive_logits(h), inp[:, t + 1], ignore_index=0)
        total += (loss / (T - 1)).item(); n += 1
    return total / n


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    train_batches, val_batches, vocab, _ = load_wikitext2(max_vocab=VOCAB, seq_len=SEQ, batch_size=BS)
    print(f"train={len(train_batches)} val={len(val_batches)}")
    model = SparsePatternLMv8(vocab_size=VOCAB, N=N, **CFG).to(device)
    print(f"cfg={CFG}")

    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        dsum, dn = 0.0, 0
        for inp, _ in tqdm(train_batches, desc=f"epoch {epoch}", leave=False):
            inp = inp.to(device)
            B, T = inp.shape
            h = model.new_state(B, device)
            for t in range(T - 1):
                h = model.integrate(h, inp[:, t])
                d = model.r_stdp(h, inp[:, t + 1])      # δ-게이팅 R-STDP
                model.homeostasis(h)
                dsum += d.item(); dn += 1
        vl = evaluate(model, val_batches, device)
        nz = (model.W.abs() > 1e-4).float().mean().item()
        print(f"epoch {epoch} | val={vl:.3f} ppl={2.718**vl:.0f} | W_nz={nz:.3f} | "
              f"theta={model.theta.mean().item():.2f} | delta={dsum/dn:.3f}")
        if vl < best:
            best = vl
            torch.save({"W": model.W, "theta": model.theta, "anchor": model.anchor, "cfg": CFG}, "model_v8_best.pt")
    print(f"best val={best:.3f} ppl={2.718**best:.0f}")


if __name__ == "__main__":
    main()
