import torch
import torch.nn as nn
import torch.nn.functional as F


class ReasoningLMv12(nn.Module):
    """
    설계 전환(#9): 입력 - 이성 - 출력 3분할. 언어=측정 probe, 사고=핵심.

    근거(Fedorenko): 뇌에서 언어망(이해/생성 I/O)과 추론망(multiple-demand)은
      분리됨. 언어 없이 사고 가능(수학·논리·실어증 추론). → 언어는 사고의 출력
      포트지 사고 자체가 아님.

    v9 문제: 이성(동역학 W)과 출력(drive_logits가 같은 W 재사용)을 conflate.
      ppl로 직접 학습 → 동역학이 lookup으로 붕괴(W 전부 음수, drive cos 0.988).

    v12 구조 (전부 뇌 대응):
      입력 anchor_in (V×N, 고정 희소)        = 이해 / Wernicke (토큰→내부표상)
      이성 W_rec (N×N, 재귀 동역학)          = multiple-demand (전역 사고)
      출력 W_out (N×V, Hebbian readout)      = 생성 / Broca (내부상태→다음토큰)

    이성 핵심 = 출력 한 토큰당 내부 n_internal 스텝 재귀 진화("말하기 전 생각").
      덧셈 leaky는 못 하지만 재귀 비선형 동역학은 다단계 계산 가능
      = 동역학이 lookup 이기는 지점.

    이성의 '자체 압력' = 내부 predictive coding (Rao-Ballard):
      재귀 연결 W_rec이 다음 입력을 예측하도록 국소 학습(delta rule).
      err = anchor_in[next] − sigmoid(h @ W_rec);  dW_rec = h.T @ err.
      → 이성은 ppl로 직접 학습되지 않음. 자기 입력을 *예측*하도록 진화.
      출력 head(W_out)가 그 내부 상태를 *읽어*낼 뿐.

    출력 학습 = Hebbian (상태 ↔ 실제 다음토큰 공동출현). 카운트가 안정(과설계 교훈).
    readout = log(relu(·)+ε)  (Weber-Fechner, 살아남은 유일한 bio readout 보정).

    측정: ppl = 출력 포트(언어망 surprisal, sanity). 동역학 가치는 별도 축
      (장거리·조합·연속학습무망각·내부구조) — 관측 훅 참고.
    """

    def __init__(
        self,
        vocab_size: int,
        N: int = 8000,
        k0: int = 10,
        n_internal: int = 4,       # 토큰당 내부 사고 스텝 ("말하기 전 생각")
        lam: float = 0.5,          # leaky 적분률 (막 시정수)
        beta: float = 4.0,
        sigma: float = 0.5,
        gamma: float = 3.0,        # 분할 정규화 (피드백 억제)
        seed_gain: float = 3.0,    # 입력 주입 이득
        rec_gain: float = 1.0,     # 재귀 흥분 이득
        rho: float = 0.06,         # 목표 발화율
        theta_init: float = 0.3,
        theta_base: float = 0.3,
        lam_theta: float = 0.01,
        eta_homeo: float = 0.02,
        lr_rec: float = 0.02,      # 이성 predictive-coding 학습률
        decay_rec: float = 0.99,   # W_rec 가지치기
        eps: float = 1e-6,
    ):
        super().__init__()
        assert N > vocab_size
        self.vocab_size = vocab_size
        self.N = N
        self.k0 = k0
        self.n_internal = n_internal
        self.lam = lam
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.seed_gain = seed_gain
        self.rec_gain = rec_gain
        self.rho = rho
        self.theta_base = theta_base
        self.lam_theta = lam_theta
        self.eta_homeo = eta_homeo
        self.lr_rec = lr_rec
        self.decay_rec = decay_rec
        self.eps = eps

        # 입력 인코더: 토큰 = 고정 무작위 k0-hot (감각 안정). 이해.
        anchor_in = torch.zeros(vocab_size, N)
        for v in range(vocab_size):
            anchor_in[v, torch.randperm(N)[:k0]] = 1.0
        self.register_buffer("anchor_in", anchor_in)

        # 이성 핵심: 희소 재귀 결합 (predictive coding으로 학습됨)
        W_rec = torch.zeros(N, N)
        mask = torch.rand(N, N) < 0.02
        mask.fill_diagonal_(False)
        W_rec[mask] = torch.randn(int(mask.sum().item())) * 0.1
        self.register_buffer("W_rec", W_rec)

        # 출력 head: 상태 → 다음토큰 (Hebbian 카운트로 채워짐, init 0)
        self.register_buffer("W_out", torch.zeros(N, vocab_size))

        self.register_buffer("theta", torch.full((N,), float(theta_init)))

    # ---- 상태 ----
    def new_state(self, B: int, device) -> torch.Tensor:
        return torch.zeros(B, self.N, device=device)

    # ---- 이성: 내부 재귀 동역학 한 스텝 ----
    def _think_step(self, h: torch.Tensor, drive: torch.Tensor | None) -> torch.Tensor:
        rec = self.rec_gain * (h @ self.W_rec)
        total = rec if drive is None else rec + self.seed_gain * drive
        activity = h.mean(dim=1, keepdim=True)
        R = total / (self.sigma + self.gamma * activity)   # 분할 정규화
        target = torch.sigmoid(self.beta * (R - self.theta.clamp(min=0.0)))
        return (1.0 - self.lam) * h + self.lam * target    # leaky 적분(지속)

    def think(self, h: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """입력 토큰 주입 + 내부 n_internal 스텝 재귀 진화. 상태 리셋 X(문맥 누적).
        = '말하기 전 생각'. 반환 = 진화된 이성 상태."""
        drive = self.anchor_in[token_ids]
        for _ in range(self.n_internal):
            h = self._think_step(h, drive)
        return h

    # ---- 출력: 생성 ----
    def speak(self, h: torch.Tensor) -> torch.Tensor:
        """이성 상태 → 다음토큰 logits. log(relu) readout."""
        scores = h @ self.W_out                            # (B, V)
        return torch.log(F.relu(scores) + self.eps)

    # ---- 학습 ----
    @torch.no_grad()
    def learn_rec(self, h: torch.Tensor, next_ids: torch.Tensor) -> torch.Tensor:
        """이성 자체 압력 = 내부 predictive coding.
        W_rec이 다음 입력 anchor를 예측하도록 국소 delta. ppl과 무관.
        반환: 평균 |예측오차|."""
        pred = torch.sigmoid(h @ self.W_rec)               # 다음 입력 예측
        target = self.anchor_in[next_ids]                  # 실제 다음 입력 (0/1)
        err = target - pred
        dW = h.t() @ err / h.shape[0]
        self.W_rec.mul_(self.decay_rec)
        self.W_rec.add_(self.lr_rec * dW)
        self.W_rec.fill_diagonal_(0.0)
        self.W_rec[self.W_rec.abs() < 1e-3] = 0.0          # 대사 가지치기
        return err.abs().mean()

    @torch.no_grad()
    def learn_out(self, h: torch.Tensor, next_ids: torch.Tensor) -> None:
        """출력 학습 = Hebbian 공동출현. 상태 h와 실제 다음토큰을 더함(카운트)."""
        self.W_out.index_add_(1, next_ids, h.t())          # W_out[:, next] += h

    @torch.no_grad()
    def homeostasis(self, h: torch.Tensor) -> None:
        rate = (h > 0.5).float().mean(dim=0)
        self.theta += self.eta_homeo * (rate - self.rho) \
                      - self.lam_theta * (self.theta - self.theta_base)
        self.theta.clamp_(min=0.0)

    # ---- 관측 훅 (동역학이 lookup 이기는 축 / 내부구조) ----
    @torch.no_grad()
    def w_sign_balance(self) -> dict:
        """W_rec 부호 균형. v9 실패=전부 음수 붕괴(W_neg≈W_nz). E/I 균형 확인."""
        nz = self.W_rec != 0
        frac_nz = nz.float().mean().item()
        frac_neg = (self.W_rec < 0).float().mean().item()
        return {"nz": frac_nz, "neg": frac_neg, "neg_of_nz": frac_neg / (frac_nz + 1e-9)}

    @torch.no_grad()
    def assembly_overlap(self, token_ids: torch.Tensor, device, free: int = 0) -> torch.Tensor:
        """관련 단어가 겹치는 이성 상태로 수렴하나(cat~dog, 원 목표).
        각 토큰 단독 think 후 상태들의 쌍별 cos. 평균 낮으면 분화(좋음),
        붕괴면 1에 가까움(v9 drive cos 0.988 = 붕괴).

        승현 진단: think()는 매 내부스텝 anchor를 재주입 → h가 무작위 직교 입력에
          지배당해 이성층 수렴을 못 잰다(입력 코드끼리 cos≈0이 상태 cos를 고정).
          free>0 이면 입력 주입 후 free 스텝 무입력 자유진행(재귀 동역학만) → 이성층
          고유 수렴을 측정. (경고: 전부음수 W_rec는 free_run서 단일 전역 attractor로
          붕괴 = sim/random 무관하게 cos→1. 그것이 진짜 실패 신호.)"""
        h = self.new_state(len(token_ids), device)
        h = self.think(h, token_ids.to(device))
        for _ in range(free):
            h = self._think_step(h, None)                  # 입력 제거, 재귀만
        hn = F.normalize(h, dim=1)
        return hn @ hn.t()                                 # (n, n) cos 행렬

    @torch.no_grad()
    def free_run(self, h: torch.Tensor, steps: int) -> torch.Tensor:
        """입력 없이 내부 동역학 자유 진행(attractor/replay probe).
        반환: 스텝별 활동률 궤적 (수렴/진동/폭발 판별)."""
        traj = []
        for _ in range(steps):
            h = self._think_step(h, None)
            traj.append((h > 0.5).float().mean().item())
        return torch.tensor(traj)
