import torch
import torch.nn as nn
import torch.nn.functional as F


class SparsePatternLMv10(nn.Module):
    """
    0단계 v10: 도파민 three-factor 규칙 (bipolar RPE × 희소 eligibility).

    v9(delta rule) 진단: dW = pre.T @ (target − pred)에서 target은 k0-hot →
        4000 중 20개만 1, 3980개 0. 매 step 침묵 노드마다 err = −pred < 0 →
        **dense LTD** → W가 구조적으로 음수 쏠림 → "전부 억제" drive 붕괴
        (analyze_v9: drive 쌍별 cos 0.988, W_neg 0.632). 폭발의 거울상.

    v8(R-STDP) 진단: δ 게이팅했으나 dW = (δ·pre).T @ anchor ≥ 0 = **LTP-only**.
        음의 학습 없음. random-data서 δ≈1 고정 → 무한 LTP → 폭발.

    v10 고침 = three-factor (Schultz 도파민 RPE):
        eligibility e = pre ⊗ post   (실현된 공동활성, post=다음토큰 anchor)
          → post=0(침묵)인 3980 노드는 e=0 → 업데이트 X → dense LTD 회피
        δ = r − V   (bipolar RPE; r=p(correct), V=EMA 기준선)
          δ>0 (예상보다 잘 맞힘) → 실현 전이 강화 (LTP)
          δ<0 (예상보다 못 맞힘) → 실현 전이 약화 (LTD)
        dW = δ · e   → 양·음 둘 다, 단 켜진 시냅스만. 수렴 시 V↑→δ→0 자기안정.
    뇌: 흑질/복측피개 도파민 RPE가 eligibility trace 태깅된 시냅스 가소성 게이팅.
        희소성이 eligibility(pre·post)에 내장 → E/I 불균형 회피.

    동역학(v4~v9 동일): leaky 적분 working memory + bounded homeostasis θ + divisive.
    """

    def __init__(
        self,
        vocab_size: int,
        N: int = 4000,
        k0: int = 10,
        lam: float = 0.5,
        n_settle: int = 4,
        beta: float = 4.0,
        sigma: float = 0.5,
        gamma: float = 3.0,
        rho: float = 0.06,
        theta_init: float = 0.3,
        theta_base: float = 0.3,
        lam_theta: float = 0.01,
        eta_homeo: float = 0.02,
        seed_gain: float = 3.0,
        stdp_lr: float = 0.02,
        decay: float = 0.99,
        rec_gain: float = 1.0,
        beta_v: float = 0.01,      # 도파민 기준선(V) EMA 속도
    ):
        super().__init__()
        assert N > vocab_size
        self.vocab_size = vocab_size
        self.N = N
        self.k0 = k0
        self.lam = lam
        self.n_settle = n_settle
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.rho = rho
        self.theta_base = theta_base
        self.lam_theta = lam_theta
        self.eta_homeo = eta_homeo
        self.seed_gain = seed_gain
        self.stdp_lr = stdp_lr
        self.decay = decay
        self.rec_gain = rec_gain
        self.beta_v = beta_v

        anchor = torch.zeros(vocab_size, N)
        for v in range(vocab_size):
            anchor[v, torch.randperm(N)[:k0]] = 1.0
        self.register_buffer("anchor", anchor)

        W = torch.zeros(N, N)
        mask = torch.rand(N, N) < 0.02
        mask.fill_diagonal_(False)
        W[mask] = torch.randn(int(mask.sum().item())) * 0.1
        self.register_buffer("W", W)

        self.register_buffer("theta", torch.full((N,), float(theta_init)))
        self.register_buffer("V", torch.tensor(1.0 / vocab_size))   # 도파민 기준선(평균보상)
        self.register_buffer("S", torch.tensor(1.0 / vocab_size))   # 보상 변동성(적응 스케일)

    def new_state(self, B: int, device) -> torch.Tensor:
        return torch.zeros(B, self.N, device=device)

    def _step(self, h: torch.Tensor, drive: torch.Tensor | None) -> torch.Tensor:
        rec = self.rec_gain * (h @ self.W)
        total = rec if drive is None else rec + self.seed_gain * drive
        activity = h.mean(dim=1, keepdim=True)
        R = total / (self.sigma + self.gamma * activity)
        target = torch.sigmoid(self.beta * (R - self.theta.clamp(min=0.0)))
        return (1.0 - self.lam) * h + self.lam * target

    def integrate(self, h: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        drive = self.anchor[token_ids]
        for _ in range(self.n_settle):
            h = self._step(h, drive)
        return h

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        s = h
        for _ in range(self.n_settle):
            s = self._step(s, None)
        return s

    def drive_logits(self, h: torch.Tensor) -> torch.Tensor:
        return (h @ self.W) @ self.anchor.t()

    @torch.no_grad()
    def r_stdp(self, pre: torch.Tensor, next_ids: torch.Tensor) -> torch.Tensor:
        """three-factor 도파민 규칙. dW = δ · (pre ⊗ anchor[next]).
        pre=문맥상태 h_t, next_ids=실제 다음토큰.
        eligibility = pre ⊗ anchor[next] (실현 공동활성, post=k0-hot 희소).
        δ = (p(correct) − V)/(S+ε) 표준화 bipolar RPE. V=평균보상 EMA,
        S=변동성 EMA(도파민 적응 스케일) → δ를 O(1)로 만들어 콜드스타트 가능.
        return: 평균 reward p(correct) (모니터링)."""
        B = pre.shape[0]
        logits = self.drive_logits(pre)                          # (B, V)
        p = torch.softmax(logits, dim=1)
        r = p[torch.arange(B, device=p.device), next_ids]        # (B,) reward=p(correct)
        delta = (r - self.V) / (self.S + 1e-8)                   # (B,) 표준화 RPE
        self.V += self.beta_v * (r.mean() - self.V)              # 평균 EMA
        self.S += self.beta_v * ((r - self.V).abs().mean() - self.S)  # 변동성 EMA

        post = self.anchor[next_ids]                             # (B, N) 실현 post (희소)
        gated_pre = pre * delta.unsqueeze(1)                     # δ 게이팅 (±)
        dW = gated_pre.t() @ post / B                            # eligibility × δ
        self.W.mul_(self.decay)
        self.W.add_(self.stdp_lr * dW)
        self.W.fill_diagonal_(0.0)
        self.W[self.W.abs() < 1e-3] = 0.0                        # 대사 가지치기
        return r.mean()

    @torch.no_grad()
    def homeostasis(self, h: torch.Tensor) -> None:
        rate = (h > 0.5).float().mean(dim=0)
        self.theta += self.eta_homeo * (rate - self.rho) \
                      - self.lam_theta * (self.theta - self.theta_base)
        self.theta.clamp_(min=0.0)
