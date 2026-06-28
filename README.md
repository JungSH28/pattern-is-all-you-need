# pattern-is-all-you-need

뇌(언어피질)에 부합하는 Sparse Hebbian 언어모델. **모든 구조가 뇌와 대응**하는 것이 설계 원칙 —
비생물학적 트릭(k-WTA, backprop 등)은 배제하고, 값 조정보다 구조 부합을 먼저 확정한다.

## 핵심 아이디어

- **단일 가중치 행렬 W가 곧 표현.** 벡터 i = `W[i,:]`. 별도 임베딩 없음.
- **토큰 = 수렴하는 k-sparse cell assembly.** 토큰 정체성은 고정 무작위 k₀-hot **anchor**로 인코딩
  (감각 안정성·근직교). 의미는 anchor가 아니라 학습된 W(전이)에 창발해야 한다.
- **흥분/억제 = W의 부호** (Dale 분리 없이). sigmoid 발화, tanh 아님.
- **지속 상태(working memory).** leaky 적분 `h ← (1-λ)·h + λ·sigmoid(β·(R−θ))`로 문맥 누적.
- **학습 = STDP(시간 비대칭) + delta rule(LTP+LTD).** 음의 학습(LTD)이 억제구조를 만들어 E/I 자기조절.

### 3계층 메모리 (뇌 대응)

| 계층 | 뇌 | 시간척도 | 코드 | 성격 |
|---|---|---|---|---|
| **핫** = 활성 패턴 | working memory | ms~초 | `h` (leaky τ) | 휘발 |
| **웜** = 수정가능 임베디드 | 빠른 시냅스 (초기 LTP) | 분~일 | `W` (delta 씀) | 가소·감쇠 |
| **콜드** = 고정 임베디드 | 공고화 시냅스 + 감각코드 | 일~영구 | `anchor` (고정) | 보호·불변 |

현재 웜→콜드 **공고화 경로**가 빠져 있음 → 다음 단계(v10) 과제. (파국적 망각 방지)

## 버전 아크 (각 단계 = 별도 파일)

| 버전 | 변경 | 결과 |
|---|---|---|
| v1/v2 | gradient + 대칭 Hebbian | **붕괴** (모든 패턴 병합, cos 0.999) |
| v3 | STDP (시간 비대칭) | 붕괴 해결(asm_cos 0.003) but 예측 텅 빔 |
| v4 | 지속상태(working memory, leaky τ) | 점화·문맥 but W 밀집 |
| v5 | drive readout `(h@W)@anchor.T` | 신호(ppl 1066, transient) but θ 폭주 |
| v6 | bounded homeostasis (θ baseline 복원) | θ 유계 but W 밀집 |
| v7 | synaptic scaling | **폭발** |
| v8 | R-STDP (도파민 δ 게이팅) | **폭발** (단 random-data 스모크, 불공정) |
| **v9** | **delta rule (LTP+LTD)** | **첫 안정.** best ppl 936, theta 유계 |

**핵심 통찰 (v3~v8 → v9):** 활동 폭발의 진짜 뿌리 = 음의 학습(LTD) 부재 → 억제구조 못 만듦 →
E/I 불균형. delta rule `dW = pre.T @ (target − sigmoid(drive))`이 LTD를 내장해 학습된 억제로
자기조절. ("학습이 음으로도 되나?" 질문이 정확히 짚음.)

## 현재 상태 (v9)

4에폭 완주: ppl 984→942→939→**936** (바닥 근접·포화), W_nz 0.749→0.618(자가 희소화), theta 0.19 유계.

**미해결 — 의미창발 실패** (`analyze_v9.py`):
- `drive 평균 쌍별 cos = 0.988` → 토큰별 출력 drive 거의 동일 = **drive 붕괴**.
- `W_neg = 0.632 ≈ W_nz` → 비영 가중치 거의 전부 음수. LTD 우세로 W가 음수로 쏠림 = **"전부 억제"** 솔루션.
- 즉 폭발(LTP 폭주)의 **거울상 실패**. ppl 이득은 빈도(unigram) 근사 수준.

→ 단일 가소성 W는 안정-가소성 딜레마를 못 넘음(폭발 아니면 붕괴). 토큰 정체성을 보호하는
별도 시간척도 계층(콜드) 또는 도파민 R-STDP(희소 local credit)가 다음 처방.

## 파일

```
model_v9.py      현 best 아키텍처 (SparsePatternLMv9)
train_v9.py      4에폭 학습 (drive readout eval, model_v9_best.pt 저장)
analyze_v9.py    의미창발 검증 (drive/logit cos, anchor 직교성, W_neg)
data.py          wikitext2 로더 (pad=0, unk=1)
model_v3~v8.py   각 단계 스냅샷 (붕괴/폭발 진단 기록)
model_v9_best.pt 학습된 체크포인트 (W, theta, anchor, cfg)
```

## 실행

```bash
python3 train_v9.py      # 학습 → model_v9_best.pt
python3 analyze_v9.py    # 의미창발/구조 검증
```

설정: `VOCAB=2000, N=4000, EPOCHS=4, BS=32, SEQ=16`,
`CFG = dict(rho=0.06, gamma=4, theta_init=0.3, k0=20, eta_homeo=0.02)`. device = MPS/CPU.

## 다음 (v10 후보)

1. **콜드-웜-핫 다중시간척도** — `W_eff = W_fast + W_slow`, 공고화로 흥분성 정체성 보호 (drive 붕괴·망각 처방).
2. **도파민 R-STDP** — three-factor `δ·pre·post`, 희소 eligibility로 dense LTD 불균형 회피.
3. 위 stage 1~2: 명시적 RPE, 4층(Wernicke/Broca).

상세 설계·근거는 LLM-Wiki `research/pattern-is-all-you-need.md` 참조.
