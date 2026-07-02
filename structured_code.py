"""
입력 코드 실험 (승현 진단): 무작위 배타 코드 vs 구조 코드 (같은 문맥=공유 차원).

승현 지적: 토큰을 단일 무작위 k0-hot로 고정하면 유사 문맥 단어끼리 겹치지 않는다
  (he~she 코드 cos=0.0)인데, 데이터의 진짜 분포 유사도는 크다(he~she 문맥 cos=0.997).
  = 입력 단계서 유사도 구조를 전부 버림 → 국소 규칙이 못 복원(#6/#8 arc와 동근).

구조 코드 = 문맥 프로파일(이전단어 분포)을 무작위 사영 후 top-k → 유사 문맥 단어가
  겹치는 활성 차원 공유. sensory cortex의 활동의존 발달(무작위X 유전고정X)에 대응.

측정: h=code[prev], W_out=Hebbian 카운트, readout=log(relu+ε)+α·unigram.
  코드 효과만 분리(동역학 없음).

결과 (실측):
  VOCAB=2000 : RANDOM 78  STRUCTURED 89   (구조가 오히려 나쁨)
  VOCAB=10000: RANDOM 486 STRUCTURED 321  (구조가 34% 우위)
결론: bigram 조밀 영역(작은 vocab)선 정확검색이 최적, 구조 코드의 일반화는 흐리기만.
  bigram 희소 영역(큰 vocab)선 무작위 코드는 미관측 쌍서 붕괴, 구조 코드가 일반화로 회복.
  = 승현 가설이 데이터 희소한 영역에서 성립. arc #8 "B는 vocab 10k+ 필요" 실측 확증.
"""
import torch
import torch.nn.functional as F

from data import load_wikitext2

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N, K0 = 8000, 10


def build(vocab_size: int):
    tb, vb, vocab, w2i = load_wikitext2(max_vocab=vocab_size, seq_len=16, batch_size=32)
    ctx = torch.zeros(vocab_size, vocab_size, device=DEV)  # ctx[w, prev]
    for inp, _ in tb:
        inp = inp.to(DEV)
        for row in inp:
            ctx.index_put_((row[1:], row[:-1]),
                           torch.ones(len(row) - 1, device=DEV), accumulate=True)
    prof = F.normalize(torch.log1p(ctx), dim=1)

    torch.manual_seed(0)
    rand = torch.zeros(vocab_size, N, device=DEV)
    for v in range(vocab_size):
        rand[v, torch.randperm(N)[:K0]] = 1
    proj = prof @ torch.randn(vocab_size, N, device=DEV)
    stru = torch.zeros(vocab_size, N, device=DEV)
    stru.scatter_(1, proj.topk(K0, dim=1).indices, 1.0)
    return tb, vb, rand, stru


@torch.no_grad()
def ppl_of(tb, vb, A, alpha=0.01):
    W_out = torch.zeros(N, A.shape[0], device=DEV)
    U = torch.zeros(A.shape[0], device=DEV)
    for inp, _ in tb:
        inp = inp.to(DEV)
        for row in inp:
            W_out.index_add_(1, row[1:], A[row[:-1]].t())
            U.index_add_(0, row[1:], torch.ones(len(row) - 1, device=DEV))
    U = torch.log(U + 1.0)
    tot, n = 0.0, 0
    for inp, _ in vb[:200]:
        inp = inp.to(DEV)
        for row in inp:
            logits = torch.log(F.relu(A[row[:-1]] @ W_out) + 1e-6) + alpha * U
            tot += F.cross_entropy(logits, row[1:], ignore_index=0).item()
            n += 1
    return 2.718 ** (tot / n)


if __name__ == "__main__":
    for vocab_size in [2000, 10000]:
        tb, vb, rand, stru = build(vocab_size)
        print(f"VOCAB={vocab_size}: RANDOM={ppl_of(tb, vb, rand):.1f}  "
              f"STRUCTURED={ppl_of(tb, vb, stru):.1f}", flush=True)
