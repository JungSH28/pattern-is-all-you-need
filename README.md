# pattern-is-all-you-need

뇌 부합(brain-aligned) sparse 언어모델 연구. **설계 원칙: 모든 구조가 뇌 기전과 대응하고, 학습은
backprop이 아닌 생물학적으로 그럴듯한 국소 규칙으로 한다.** 목표는 transformer의 ppl을 이기는 것이
아니라, (1) sparse recurrent/feedforward assembly로 언어를 처리하는 뇌 부합 substrate를 만들고
(2) 그 위에 언어 head를 얹어 **간단한 대화가 가능한 수준**까지 도달하는 것.

> 상세한 과정·근거·기각한 길은 위키 `research/pattern-is-all-you-need.md`(#1~#18) 참조. 이 README는
> 현 상태와 재현법의 요약이다.

---

## 1. 논지 (Thesis)

- **토큰 = sparse cell assembly.** 각 토큰은 고정 무작위 k-hot 활성 패턴(감각 안정성). 의미는 anchor가
  아니라 학습된 연결/표현에 창발해야 한다.
- **학습 = 국소 규칙.** backprop의 weight transport(순전파 가중치를 거꾸로 읽기)는 뉴런이 못 한다.
  대신 feedback alignment 계열(고정 무작위 피드백)로 credit을 전달한다.
- **언어 ≠ 사고.** 언어망(이해/생성 I/O)과 추론망(multiple-demand)은 분리된다 (Fedorenko et al.).
  → 입력(이해) · 이성(조합/연산) · 출력(생성)을 분리.
- **효율이 우위 축.** ppl 천장은 transformer가 낮을 수 있으나, sparse+국소 학습은 **같은 계산당 성능
  (학습 효율)·저활성률·online 학습·해석 가능성**에서 이길 여지가 있다.

---

## 2. 현재 아키텍처

```
입력(정체성)     이성(조합/라우팅)              출력(생성)
sparse code  →   fixed-random routing      →   Hebbian/local readout
(k-hot,          (reservoir attention,          (log readout,
 또는 학습        mixed selectivity)             unigram backoff)
 임베딩)          ★학습 불필요, 고정 무작위★
```

- **입력 코드**: k-hot sparse. 무작위(감각 안정) 또는 문맥 프로파일 기반 **구조 코드**(유사 문맥 토큰이
  차원 공유 → 일반화). 큰 vocab에서 구조 코드가 유의미하게 우수.
- **이성/라우팅**: 혼합선택성 확장 또는 dot-product 라우팅. **핵심 발견 — 라우팅 사영을 고정 무작위
  (reservoir; Maass, Jaeger)로 둔 게 학습된 라우팅보다 낫다.** 조합(t-1,t binding)은 순서보존 사영
  (P1≠P2) + 높은 threshold(AND) 또는 softmax 라우팅.
- **출력**: `log(relu(·)+ε) + α·unigram`. Weber-Fechner 로그 압축 + 기저 흥분성 backoff.
- **학습**: 국소 delta(readout) + feedback alignment/DFA(임베딩). Adam/backprop은 상한 비교용 발판.

---

## 3. 버전 아크 (요약)

각 단계는 별도 파일. 상세 진단은 위키.

| 버전 | 핵심 | wikitext2 vocab2000 val ppl |
|---|---|---|
| v1/v2 | gradient + 대칭 Hebbian | 붕괴 (모든 패턴 병합) |
| v3~v8 | STDP·working memory·항상성·R-STDP | 붕괴/폭발 반복 |
| v9 | delta rule (LTP+LTD, 음의 학습) | 936 (첫 안정, 단 의미창발 실패) |
| v11 | 순수 Hebbian 카운트 + log readout | 348 |
| — | 카운트 bigram + unigram backoff | **63** |
| v12 | 3분할 동역학(재귀 이성) | 단일 attractor 붕괴 (정체성 소실) |
| v13 | 이성 = 조합(mixed selectivity), 정적 | vocab10k: 무작위 486 → 구조 311 → +조합 302.7 |
| v14 | 학습 임베딩 + 비선형 조합 | backprop **55.4** / bio(sign) 64.5 / FA 69 |
| v15 | attention 레버 규명 | additive 61.8 → attention 58.6 → frozen-routing 56.8 |
| — | (참고) 동일크기 transformer | 0.42M **53**, 0.9M **44** |

---

## 4. 핵심 발견 (진단 사슬)

1. **붕괴 2종 분리.** ①부호붕괴(all-negative): predictive-coding target이 mostly-0이라 오차가 전부
   음수 → vocab 무관. **err-centering**(오차 zero-mean; cross-entropy 기울기에 내장)으로 해결. ②attractor
   붕괴(단일 고정점): 재귀 동역학에 vocab을 다 저장 = Hopfield 용량 초과(용량 ~0.14·N ≪ vocab). 정체성을
   fixed-point에 두는 전제가 실패 → 정체성은 입력 코드에, 이성은 조합/연산으로 재정의.
2. **표현학습은 bio-국소로 가능.** feedback alignment(backprop 없음)가 임베딩을 학습하고 붕괴하지 않는다
   (vocab2000: FA 83→ 하락, backprop 62.9). 콜드-웜-핫 scaffold(구조 prior + FA 학습)가 무작위 init보다
   우수. → "credit assignment = 하드월"은 반증됨. 남은 건 성능 격차.
3. **구조화 피드백 > 무작위.** hidden 피드백: random-FA 70 < sign-concordant 64.5 < backprop 60.4.
   부호 일치(E/I 일관) 피드백이 격차 대부분을 메운다. 순수 무작위는 하한 증명이고, 실제 뇌는 구조화
   피드백(predictive coding 방향)을 쓸 것.
4. **조합(binding)이 남은 레버.** 덧셈 결합은 순서/상호작용을 못 잡아 천장(58). softmax 라우팅
   (attention)이 additive를 이긴다(58.6 vs 61.8). 다만 pairwise Hadamard(곱을 feature로)는 실패
   (product를 *score*로 써야). 완전 폐쇄(→53)는 transformer 스택 필요 = 비-bio.
5. **간단한 대화 달성.** 큰 vocab(→ `<unk>` 홍수 제거) + 대화 데이터(empathetic dialogues, 턴마커)로
   frozen-routing 모델이 공감 레지스터로 응답한다. 완전-bio(DFA) 버전도 작동(품질 비용 있음).

---

## 5. 평가 (표준 지표 + 효율)

**대화 held-out val perplexity** (empathetic_dialogues, ~1.15M params, 동일 크기):

| 모델 | 학습 | val ppl |
|---|---|---|
| Ours (frozen-attn, backprop E/head) | 5 epoch (수렴) | **96.8** |
| 동일크기 transformer | 10 / 20 / 25 epoch | 128.6 / 92.9 / 87.2 |
| Ours (fully-bio, DFA) | 15 epoch | 160.5 (하락중) |

- **효율 우위:** ours는 5 epoch에 96.8 도달, transformer는 20 epoch 필요(≈4~5× 학습 효율). ppl *천장*은
  transformer가 낮으나(87.2), *같은 계산당 성능*은 ours가 앞선다 — "GPT 낭비 우회" 논지와 연결.
- **완전-bio 비용:** DFA(backprop 없음)는 160(15ep, ~1.7× 격차, 하락중), 공감 레지스터는 유지하나
  backprop보다 거침(`<unk>`·반복↑). 격차 좁히기 두 시도 모두 실패: (a) 구조화
  피드백(sign-concordant, 직접 경로) 169.8 ≈ random 169.1; (b) attention-mix 경로까지 credit 라우팅
  (full-path DFA) 199.1 — *오히려 악화*(무작위 피드백 행렬을 더할수록 신호 아닌 노이즈 증가). → **격차는
  "credit 경로 누락"이 아니라 random-feedback credit의 본질적 근사 한계.** feedback 조정으로는 못 좁히며,
  근본적으로 더 나은 bio credit(predictive coding·target propagation 등, 분야 미제)이 필요.
- **지표 한계:** ppl은 lookup 편향 지표(count 표가 동역학을 이긴다). sanity check용이며, 사고/동역학 가치는
  별도 축(장거리·조합·연속학습·내부구조 RSA)이 필요.

**효율 정직한 정량화** (`efficiency.py`, `dialogue_sparse.py`):
- per-token FLOP: 우리 822k ≈ tf 1.17M (둘 다 vocab readout head가 지배, dense면 큰 차이 없음).
- 진짜 우위: ①학습 파라미터 적음(라우팅 고정) ②4~5× 빠른 수렴 ③backprop 없음(DFA, forward 위주).
- **sparsity 복원(중요):** 대화모델은 학습 dense 임베딩을 써서 density 0.69였다(sparse 원칙 이탈). feat에
  top-k 강제로 sparse 복원: density 0.06(k=8/128, 뇌 발화율 ~5% 수준)에서도 대화 유지, 비용은 ppl
  105→227(~2.2×). feat 6% 활성 → 지배적 head 계산 ~11× 절감 = sparse-coding 에너지/정확도 tradeoff
  (뉴로모픽 가치). 효율 우위는 dense가 아니라 **sparse 강제 + 학습효율**에서 나온다.
- **sparse를 유전 prior로 + threshold 발화(`dialogue_adaptive_sparse.py`):** 하드 top-k 대신 **threshold
  발화**(입력>θ, θ는 항상성으로 목표 발화율 ρ 유지) = 뉴런식 발화 + 자연 변동 k. density 0.10, ppl 207 —
  **고정 top-k(228)보다 우수**(강한 유닛은 살리고 약한 것만 죽임). k는 입력마다 변동(std 0.06)하나 난이도와
  상관 없음(+0.015). = 설계 원칙 "sparse는 학습으로 창발하지 않고 유전 prior(항상성 threshold)로 부여"의 구현.

**대화 샘플** (backprop 모델, `gen_saved.py`):

```
USER: my dog is sick
BOT : i would be so upset about it.
USER: i am scared about my exam
BOT : oh wow! i hate that is so i hope he is very painful.
```

---

## 6. 방법론 원칙

- **진단 먼저, 메커니즘 나중.** 아이디어를 구현하기 전에 실패의 뿌리를 실측으로 규명 (이 프로젝트의
  반복된 교훈; 정교한 학습규칙 대부분이 기본 통계를 부수었다).
- **bio = 발판.** 구조를 뇌 부합으로 되게 만든 뒤, 상한 확인엔 backprop을 써도 된다. 최종 목표가 bio-국소.
- **소스 직접 검증.** 성능 수치·버그는 코드를 돌려 확인 (예: MPS `multinomial`이 범위 밖 인덱스를 반환하는
  버그, 인덱싱 off-by-one, val subset ~1.5% 낙관 편향 — 전부 실행으로 잡음).

---

## 7. 한계 · 열린 문제

- **bio-국소 attention.** 라우팅 이점을 고정 무작위(reservoir)로 잡았으나 softmax-dot은 완전 bio가 아님.
  뇌의 binding(theta-gamma 위상부호, 수상돌기 동시검출, 혼합선택성)로의 대체가 미해결.
- **동역학 복귀.** 재귀 이성(v12)은 attractor 용량 문제로 보류. 다중 시간척도(Benna-Fusi)·공고화가 처방
  후보.
- **신경조절(도파민) 미적용.** three-factor(pre×post×도파민 = 보상/오차 게이팅)는 feedforward 학습에도
  붙는다(동역학 불필요). 대화 품질(bio-RLHF)의 다음 후보.
- **DFA 성능 격차.** 완전-bio 학습이 backprop 대비 2× ppl. 구조화 피드백·predictive coding으로 좁히기.

---

## 8. 참조 문헌 (References)

**국소 학습 / credit assignment**
- Hebb, D.O. (1949). *The Organization of Behavior.* — Hebbian 학습, cell assembly.
- Bi, G. & Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons. *J. Neurosci.* — STDP.
- Lillicrap, T. et al. (2016). Random synaptic feedback weights support error backpropagation. *Nat. Commun.* — **Feedback Alignment**.
- Nøkland, A. (2016). Direct Feedback Alignment provides learning in deep neural networks. *NeurIPS* — **DFA**.
- Liao, Q., Leibo, J., Poggio, T. (2016). How important is weight symmetry in backpropagation? *AAAI* — sign-concordant feedback.
- Kolen, J. & Pollack, J. (1994). Backpropagation without weight transport. *IEEE ICNN*; Akrout et al. (2019) — weight mirror.
- Guerguiev, J., Lillicrap, T., Richards, B. (2017). Towards deep learning with segregated dendrites. *eLife* — 수상돌기 오차.
- Payeur, A. et al. (2021). Burst-dependent synaptic plasticity. *Nat. Neurosci.*

**희소/혼합선택성/리저버**
- Maass, W., Natschläger, T., Markram, H. (2002). Real-time computing without stable states (Liquid State Machines). *Neural Comput.*
- Jaeger, H. (2001). Echo State Networks. — reservoir computing.
- Rigotti, M. et al. (2013). The importance of mixed selectivity in complex cognitive tasks. *Nature.*
- Marr, D. (1969); Albus, J. (1971) — 소뇌 granule cell 확장 코드 / 지도학습.
- Quian Quiroga, R. et al. (2005). Invariant visual representation by single neurons (concept cells). *Nature.*

**예측/신경조절/항상성**
- Rao, R. & Ballard, D. (1999). Predictive coding in the visual cortex. *Nat. Neurosci.*
- Friston, K. (2005). A theory of cortical responses. *Phil. Trans. R. Soc.*
- Schultz, W., Dayan, P., Montague, P. (1997). A neural substrate of prediction and reward (dopamine RPE). *Science.*
- Benna, M. & Fusi, S. (2016). Computational principles of synaptic memory consolidation. *Nat. Neurosci.* — 다중 시간척도.
- Turrigiano, G. (2008). The self-tuning neuron: synaptic scaling / homeostasis. *Cell.*

**언어·뇌 지표**
- Fedorenko, E. et al. — 언어망 vs multiple-demand(추론)망 분리.
- Schrimpf, M. et al. (2021). The neural architecture of language: integrative modeling. *PNAS* — surprisal ↔ 언어망 fMRI/N400.
- Lisman, J. & Jensen, O. (2013). The theta-gamma neural code. *Neuron.*

**비교 대상**
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS* — transformer(제목 오마주).
- Merity, S. et al. (2016). WikiText-2. — 벤치 데이터.
- Rashkin, H. et al. (2019). EmpatheticDialogues. — 대화 데이터.

---

## 9. 파일 · 재현

```
model_v9.py / v12.py / v13.py    아키텍처 스냅샷 (붕괴/조합 진단 기록)
train_v13.py                     v13 학습/평가 (structured code + 조합 + Hebbian)
structured_code.py               구조 코드 실험 (#11)
data.py                          wikitext2 로더 (pad=0, unk=1)
```

> 이번 세션의 실험 스크립트(v14/v14b/v14ctx/v15/v15b/v15c, replearn/rl2, 대화 dlg/gen_saved/dlg_fa,
> 벤치 dlg_bench/fairtf)는 재현용으로 정리 예정. 대화 모델 체크포인트: `dlg_model.pt`.

```bash
python3 train_v13.py     # 조합 이성 (vocab10k)
# 대화: empathetic_dialogues 학습 → 응답 생성 (dlg 스크립트)
```

핵심 좌표 (vocab2000): `uniform 2000 | unigram 125 | bigram 78 | count+backoff 63 |
v14-backprop 55 | v14-bio(sign) 64 | attention 58 | tf 53 | tf-0.9M 44`.
대화(vocab6000, 1.15M): `ours 96.8(5ep) | tf 87(25ep) | ours-DFA 192(8ep, 완전 bio)`.
