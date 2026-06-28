"""
0단계 v10 — 도파민 three-factor (bipolar RPE × 희소 eligibility).
문맥상태 h_t → 실현 다음토큰 anchor로, δ=p(correct)−V 게이팅(양·음).
v9 dense-LTD 음수쏠림(drive 붕괴) 회피 목표. W_neg 모니터.
"""
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import load_wikitext2
from model_v10 import SparsePatternLMv10

VOCAB, N = 2000, 4000
EPOCHS, BS, SEQ = 4, 32, 16
CFG = dict(rho=0.06, gamma=4, theta_init=0.3, k0=20, eta_homeo=0.02, stdp_lr=0.05, decay=0.99)


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
    model = SparsePatternLMv10(vocab_size=VOCAB, N=N, **CFG).to(device)
    print(f"cfg={CFG}")

    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        rsum, rn = 0.0, 0
        for inp, _ in tqdm(train_batches, desc=f"epoch {epoch}", leave=False):
            inp = inp.to(device)
            B, T = inp.shape
            h = model.new_state(B, device)
            for t in range(T - 1):
                h = model.integrate(h, inp[:, t])
                r = model.r_stdp(h, inp[:, t + 1])      # three-factor 도파민
                model.homeostasis(h)
                rsum += r.item(); rn += 1
        vl = evaluate(model, val_batches, device)
        nz = (model.W.abs() > 1e-4).float().mean().item()
        neg = (model.W < 0).float().mean().item()
        if vl < best:
            best = vl
            torch.save({"W": model.W, "theta": model.theta, "anchor": model.anchor,
                        "V": model.V, "cfg": CFG}, "model_v10_best.pt")
        ppl = math.exp(vl) if vl < 700 else float("inf")
        print(f"epoch {epoch} | val={vl:.3f} ppl={ppl:.0f} | W_nz={nz:.3f} W_neg={neg:.3f} | "
              f"theta={model.theta.mean().item():.2f} | reward={rsum/rn:.4f} V={model.V.item():.4f}")
    bppl = math.exp(best) if best < 700 else float("inf")
    print(f"best val={best:.3f} ppl={bppl:.0f}")


if __name__ == "__main__":
    main()
