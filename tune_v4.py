"""
v4 하이퍼파라미터 sweep — sparse-ignited regime 탐색.
짧은 부분학습(N_BATCH) 후 W_nz / pred_active / ppl 측정.
목표: pred_active 20~60 (점화 유지) + W_nz < 0.3 (희소) + ppl < 2000.
"""
import torch
import torch.nn.functional as F

from data import load_wikitext2
from model_v4 import SparsePatternLMv4

VOCAB, N = 2000, 4000
N_BATCH = 400      # 부분 학습 배치 수
M_VAL   = 30       # eval 배치 수

CONFIGS = [
    dict(rho=0.01,  gamma=5, theta_init=0.3,  decay=0.99, k0=20, eta_homeo=0.05),
    dict(rho=0.02,  gamma=5, theta_init=0.3,  decay=0.99, k0=15, eta_homeo=0.05),
    dict(rho=0.005, gamma=6, theta_init=0.4,  decay=0.99, k0=20, eta_homeo=0.05),
    dict(rho=0.02,  gamma=4, theta_init=0.25, decay=0.99, k0=20, eta_homeo=0.05),
    dict(rho=0.01,  gamma=6, theta_init=0.4,  decay=0.98, k0=20, eta_homeo=0.08),
]


@torch.no_grad()
def quick_eval(model, val_batches, device, m):
    total, n, act = 0.0, 0, 0.0
    for inp, _ in val_batches[:m]:
        inp = inp.to(device)
        B, T = inp.shape
        h = model.new_state(B, device)
        loss = 0.0
        for t in range(T - 1):
            h = model.integrate(h, inp[:, t])
            pred = model.predict(h)
            loss += F.cross_entropy(model.readout(pred), inp[:, t + 1], ignore_index=0)
            act += (pred > 0.5).float().sum(1).mean().item()
        total += (loss / (T - 1)).item(); n += 1
    return total / n, act / (n * (T - 1))


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    train_batches, val_batches, _, _ = load_wikitext2(max_vocab=VOCAB, seq_len=16, batch_size=32)

    print(f"\n{'cfg':30s} | W_nz  | pred_act | ppl")
    print("-" * 60)
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
        vl, act = quick_eval(model, val_batches, device, M_VAL)
        nz = (model.W.abs() > 1e-4).float().mean().item()
        tag = f"rho{cfg['rho']} g{cfg['gamma']} th{cfg['theta_init']} dec{cfg['decay']} k{cfg['k0']}"
        print(f"{tag:30s} | {nz:.3f} | {act:7.1f}  | {2.718**vl:.0f}")


if __name__ == "__main__":
    main()
