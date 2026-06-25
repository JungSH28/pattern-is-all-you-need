"""
v5 — 예측 readout을 drive(h@W)로. model_v4 동역학 그대로, eval만 변경.
predict(입력없는 forward)가 희소 θ에서 죽는 문제 우회:
  logits = (h @ W) @ anchor.T   (threshold 안 거친 전이 drive)
"""
import torch
import torch.nn.functional as F

from data import load_wikitext2
from model_v4 import SparsePatternLMv4

VOCAB, N = 2000, 4000
N_BATCH = 400
M_VAL   = 30

CONFIGS = [
    dict(rho=0.02,  gamma=5, theta_init=0.3,  decay=0.99, k0=20, eta_homeo=0.05),
    dict(rho=0.01,  gamma=6, theta_init=0.4,  decay=0.98, k0=20, eta_homeo=0.08),
    dict(rho=0.03,  gamma=4, theta_init=0.25, decay=0.99, k0=20, eta_homeo=0.05),
]


@torch.no_grad()
def quick_eval(model, val_batches, device, m):
    """drive readout: logits = (h@W) @ anchor.T."""
    total, n = 0.0, 0
    for inp, _ in val_batches[:m]:
        inp = inp.to(device)
        B, T = inp.shape
        h = model.new_state(B, device)
        loss = 0.0
        for t in range(T - 1):
            h = model.integrate(h, inp[:, t])
            drive = h @ model.W                       # 다음으로 미는 양
            logits = drive @ model.anchor.t()
            loss += F.cross_entropy(logits, inp[:, t + 1], ignore_index=0)
        total += (loss / (T - 1)).item(); n += 1
    return total / n


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    train_batches, val_batches, _, _ = load_wikitext2(max_vocab=VOCAB, seq_len=16, batch_size=32)

    print(f"\n{'cfg':28s} | W_nz  | ppl(drive)")
    print("-" * 48)
    for cfg in CONFIGS:
        torch.manual_seed(0)
        model = SparsePatternLMv4(vocab_size=VOCAB, N=N, **cfg).to(device)
        for inp, _ in train_batches[:N_BATCH]:
            inp = inp.to(device)
            B, T = inp.shape
            h = model.new_state(B, device)
            prev = None
            for t in range(T):
                h = model.integrate(h, inp[:, t])
                if prev is not None:
                    model.stdp(prev, h)
                model.homeostasis(h)
                prev = h
        vl = quick_eval(model, val_batches, device, M_VAL)
        nz = (model.W.abs() > 1e-4).float().mean().item()
        tag = f"rho{cfg['rho']} g{cfg['gamma']} th{cfg['theta_init']} k{cfg['k0']}"
        print(f"{tag:28s} | {nz:.3f} | {2.718**vl:.0f}")


if __name__ == "__main__":
    main()
