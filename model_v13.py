import torch
import torch.nn as nn
import torch.nn.functional as F


class ReasoningLMv13(nn.Module):
    """
    설계 전환 (#12 → v13): 이성 = 조합연산(conjunction). (B안, 승현 선택)

    v12까지의 벽: 이성을 "정체성 보존 attractor"로 두려 함 → 단일 고정점 붕괴
      (Hopfield 용량초과: 용량 ~0.14·N ≪ vocab). fixed-point는 정체성 못 담음.

    v13 재정의 (전부 뇌 대응):
      입력(정체성) = structured anchor code   = Wernicke/감각. 정체성은 여기 있음.
      이성(조합)   = mixed-selectivity 확장   = 연합피질/소뇌 granule. (t-1,t) 결합.
      출력(생성)   = Hebbian readout          = Broca. 상태→다음토큰.

    핵심 = 이성층은 정체성 보존이 아니라 *순서보존 조합*을 함(arc #5~#8이 지목한 진짜
      레버, attention 이기는 지점). 덧셈 leaky는 교환법칙→순서소실이라 불가(#5).
      해법 = 서로 다른 사영 P1·P2 + 높은 threshold:
        r = relu(P1·c_t + P2·c_{t-1} − θ)
      P1≠P2 → 순서보존. 높은 θ → 일치검출(AND, 조합) not 합(OR, additive).

    정체성 = structured code (문맥 프로파일 기반, 유사 토큰이 차원 공유 → 일반화).
      무작위 배타 코드는 유사도 0(he~she cos 0.0)이라 희소 영역서 붕괴(#11 실측).

    readout = log(relu(Wbi 직접bigram) + w·relu(Wco 조합) + α·unigram).
      조합은 확장 손실로 노이즈 → 소량(w~0.3)으로 bigram에 *더함*(backoff).
      Weber-Fechner log + unigram backoff = 살아남은 bio readout 보정(#4).

    실측 (wikitext2 vocab10k, val ppl): 무작위코드 486 → 구조코드 bigram 311 →
      +조합(w=0.3) 302.7. 조합 AND(325) ≫ additive OR(386). arc 최초 bio-국소 조합
      NET 돌파. (vocab2000은 bigram 조밀=조합신호 없어 손해 — 큰 vocab 필요, #8.)

    학습 파라미터 ≈ 0: 고정 무작위 P1·P2 + Hebbian 카운트(Wbi/Wco/U). backprop 없음.
    """

    def __init__(
        self,
        vocab_size: int,
        n_in: int = 4000,      # 입력 코드 차원
        m: int = 6000,         # 이성(조합) 확장 차원
        k0: int = 10,          # 코드 희소도 (k0-hot)
        proj_density: float = 0.05,
        theta_conj: float = 3.0,   # 조합 threshold (높을수록 AND)
        mix_w: float = 0.3,        # 조합 혼합 가중 (bigram에 더함)
        alpha: float = 0.01,       # unigram backoff
        eps: float = 1e-6,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_in, self.m, self.k0 = n_in, m, k0
        self.theta_conj, self.mix_w, self.alpha, self.eps = theta_conj, mix_w, alpha, eps

        # 이성 확장 사영 (고정 무작위 희소, 서로 달라 순서보존)
        def sparse_proj():
            return (torch.rand(m, n_in) < proj_density).float() * torch.randn(m, n_in)
        self.register_buffer("P1", sparse_proj())
        self.register_buffer("P2", sparse_proj())

        # 채워지는 버퍼 (fit에서)
        self.register_buffer("code", torch.zeros(vocab_size, n_in))   # structured 정체성 코드
        self.register_buffer("W_bi", torch.zeros(n_in, vocab_size))   # 직접 bigram Hebbian
        self.register_buffer("W_co", torch.zeros(m, vocab_size))      # 조합 Hebbian
        self.register_buffer("U", torch.zeros(vocab_size))            # unigram (log)

    # ---- 이성: 순서보존 조합 확장 ----
    def conjoin(self, c_cur: torch.Tensor, c_prev: torch.Tensor) -> torch.Tensor:
        """r = relu(P1·c_t + P2·c_{t-1} − θ). AND 일치검출 = 조합세포."""
        return F.relu(c_cur @ self.P1.t() + c_prev @ self.P2.t() - self.theta_conj)

    # ---- 정체성 코드 구축 (문맥 프로파일 → 구조 코드) ----
    @torch.no_grad()
    def build_code(self, train_batches, device) -> None:
        V = self.vocab_size
        ctx = torch.zeros(V, V, device=device)
        for inp, _ in train_batches:
            inp = inp.to(device)
            for row in inp:
                ctx.index_put_((row[1:], row[:-1]),
                               torch.ones(len(row) - 1, device=device), accumulate=True)
        prof = F.normalize(torch.log1p(ctx), dim=1)     # 문맥(이전단어) 분포 프로파일
        proj = prof @ torch.randn(V, self.n_in, device=device)
        code = torch.zeros(V, self.n_in, device=device)
        code.scatter_(1, proj.topk(self.k0, dim=1).indices, 1.0)  # 유사문맥→공유차원
        self.code.copy_(code)

    # ---- 출력 학습: Hebbian 카운트 (bigram + 조합 + unigram) ----
    @torch.no_grad()
    def fit(self, train_batches, device) -> None:
        self.build_code(train_batches, device)
        for inp, _ in train_batches:
            inp = inp.to(device)
            for row in inp:
                c_cur, c_prev, nxt = self.code[row[1:-1]], self.code[row[:-2]], row[2:]
                self.W_bi.index_add_(1, nxt, c_cur.t())
                self.W_co.index_add_(1, nxt, self.conjoin(c_cur, c_prev).t())
                self.U.index_add_(0, nxt, torch.ones(len(nxt), device=device))
        self.U.copy_(torch.log(self.U + 1.0))

    # ---- 생성: 혼합 log readout ----
    @torch.no_grad()
    def logits(self, prev_ids: torch.Tensor, cur_ids: torch.Tensor) -> torch.Tensor:
        """(t-1, t) → 다음토큰 logits. bigram + w·조합 + α·unigram, log 압축."""
        c_cur, c_prev = self.code[cur_ids], self.code[prev_ids]
        bi = F.relu(c_cur @ self.W_bi)
        co = F.relu(self.conjoin(c_cur, c_prev) @ self.W_co)
        return torch.log(bi + self.mix_w * co + self.eps) + self.alpha * self.U
