"""
v12 학습/관측 스캐폴드 — 입력-이성-출력 3분할.
  학습: 이성 W_rec = 내부 predictive coding (자체 압력),
        출력 W_out = Hebbian (상태→다음토큰).
  측정: ppl(언어망 probe, sanity) + 동역학 관측(부호균형/assembly/free_run).
실행은 나중에 (승현). N 키우면 dense N×N=느림·발열 주의(~12~20k 상한).
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import load_wikitext2
from model_v12 import ReasoningLMv12

VOCAB, N = 2000, 8000
EPOCHS, BS, SEQ = 4, 32, 16
CFG = dict(n_internal=4, rho=0.06, gamma=3.0, k0=10, lr_rec=0.02)
# 관측용 단어쌍 (cat~dog 수렴 확인). 학습 후 vocab 인덱스로 매핑.
PROBE_WORDS = ["king", "queen", "man", "woman", "cat", "dog", "paris", "france"]


@torch.no_grad()
def evaluate(model, val_batches, device):
    total, n = 0.0, 0
    for inp, _ in val_batches:
        inp = inp.to(device)
        B, T = inp.shape
        h = model.new_state(B, device)
        loss = 0.0
        for t in range(T - 1):
            h = model.think(h, inp[:, t])
            loss += F.cross_entropy(model.speak(h), inp[:, t + 1], ignore_index=0)
        total += (loss / (T - 1)).item(); n += 1
    return total / n


@torch.no_grad()
def observe(model, vocab, w2i, device):
    """동역학 관측 — ppl이 못 보는 축."""
    sign = model.w_sign_balance()
    ids = torch.tensor([w2i[w] for w in PROBE_WORDS if w in w2i])
    rep = "n/a"
    if len(ids) >= 2:
        cos = model.assembly_overlap(ids, device)
        off = cos[~torch.eye(len(ids), dtype=torch.bool)]
        rep = f"assembly cos mean={off.mean():.3f} (붕괴=1)"
    h0 = model.new_state(1, device)
    h0 = model.think(h0, ids[:1].to(device)) if len(ids) else h0
    traj = model.free_run(h0, steps=20)
    print(f"  [관측] W_rec nz={sign['nz']:.3f} neg_of_nz={sign['neg_of_nz']:.3f} "
          f"(전부음수붕괴=1) | {rep} | free_run 활동 {traj[0]:.3f}→{traj[-1]:.3f}")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    train_batches, val_batches, vocab, w2i = load_wikitext2(
        max_vocab=VOCAB, seq_len=SEQ, batch_size=BS
    )
    print(f"train={len(train_batches)} val={len(val_batches)} N={N}")
    model = ReasoningLMv12(vocab_size=VOCAB, N=N, **CFG).to(device)
    print(f"cfg={CFG}")

    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        rec_err = 0.0; cnt = 0
        for inp, _ in tqdm(train_batches, desc=f"epoch {epoch}", leave=False):
            inp = inp.to(device)
            B, T = inp.shape
            h = model.new_state(B, device)
            for t in range(T - 1):
                h = model.think(h, inp[:, t])           # 이성: 내부 사고
                e = model.learn_rec(h, inp[:, t + 1])   # 이성 학습: predictive coding
                model.learn_out(h, inp[:, t + 1])       # 출력 학습: Hebbian
                model.homeostasis(h)
                rec_err += e.item(); cnt += 1
        vl = evaluate(model, val_batches, device)
        ppl = 2.718 ** vl if vl < 700 else float("inf")
        print(f"epoch {epoch} | val={vl:.3f} ppl={ppl:.0f} | rec_err={rec_err/cnt:.3f}")
        observe(model, vocab, w2i, device)
        if vl < best:
            best = vl
            torch.save(
                {"W_rec": model.W_rec, "W_out": model.W_out,
                 "anchor_in": model.anchor_in, "theta": model.theta, "cfg": CFG},
                "model_v12_best.pt",
            )
    print(f"best ppl={2.718 ** best:.0f}")


if __name__ == "__main__":
    main()
