# pattern-is-all-you-need

토큰·관념을 벡터/유닛 집합의 활성 패턴으로 표현하는 sparse 언어모델 연구다. 최종 목표는 이 표현을 실제
대화에서 사용하는 모델이며, 기능 본선과 생물 국소 본선을 구분해 함께 개발한다. 생물 본선은 backprop에
의존하지 않고 연결·학습·기억이 국소적으로 작동해야 하며, 막힌 계산의 돌파구를 생명 기전에서 찾는다.
**간단한 대화**는 최종 정의가 아니라 첫 외부 이정표다.

> 상세한 과정·근거·기각한 길은 위키 `research/pattern-is-all-you-need.md`(#1~#29) 참조. 이 README는
> 현 상태와 재현법의 요약이다.

### 현재 연구 방향

사용자 원문:

> "일단 방향을 토큰을 벡터 집합으로 인코딩해서 대화하며 사용 가능한 모델 + 생물 국소 모델에서 대화하며
> 사용 가능한 모델(돌파구를 생명 기전에서 착안)로 하자"

두 트랙이 공유하는 불변은 **토큰·관념을 단일 embedding 행이 아니라 벡터/유닛 assembly로 인코딩하고,
문맥·범주·기억·사고가 그 연결과 활성에서 작동해 대화로 드러나는 것**이다.

1. **기능 본선**: 이 표현을 유지하면서 실제 대화에 사용할 수 있는 모델을 만든다. 필요한 공학적 학습법을
   사용할 수 있지만, 어떤 비국소 장치가 성능에 기여했는지 분리해 기록한다.
2. **생물 국소 본선**: 동일한 대화 능력을 국소 시냅스 정보와 sparse 동역학으로 구현한다. 전역 오차,
   전역 top-k, 전역 weakest-edge 검색 같은 scaffold는 생물학적 성공으로 세지 않고 하나씩 국소 기전으로
   교체한다.
3. **돌파구 원칙**: 생물 본선이 막히면 attention/backprop을 그대로 숨겨 넣지 않는다. 국소 억제,
   dendritic feedback, structural plasticity, replay, consolidation, neuromodulation 등 생명 기전에서 계산
   가설을 만들고 기능·절제 실험으로 검증한다.
4. **평가 원칙**: 기능 본선의 대화 품질과 생물 본선의 국소성·대화 품질을 별도 표로 측정한다. 기능 모델의
   성공을 생물 모델의 성공으로 합산하지 않는다.

### 목표의 층위

1. **기질(substrate)**: 입력·저장/이성·출력 유닛이 하나의 sparse 연결망을 이루고, 토큰·관념은 유닛
   assembly로 표현된다.
2. **내부 능력**: 문맥에 따른 다른 상태 전개, 범주 공유, 장기기억 공고화, 순서·조합·사고를 각각 직접
   probe한다.
3. **언어 행동**: 위 능력을 사용해 내용에 맞는 대화를 생성한다. 유창한 공감 문장만 만드는 것은 성공으로
   세지 않는다.

---

## 1. 논지 (Thesis)

- **토큰·관념 = sparse cell assembly.** 감각 입력은 안정된 sparse seed를 줄 수 있지만, 저장·이성
  substrate의 assembly와 의미 관계는 연결 및 활동에서 형성되어야 한다.
- **두 학습 트랙.** 기능 본선은 assembly 표현을 유지한 채 대화 가능성의 상한과 병목을 찾는다. 생물
  본선은 backprop의 weight transport 없이 국소 규칙으로 같은 능력을 재현하며, feedback alignment도
  최종 해답이 아니라 비교할 공학적 scaffold로 취급한다.
- **언어 ≠ 사고.** 언어망(이해/생성 I/O)과 추론망(multiple-demand)은 분리된다 (Fedorenko et al.).
  → 입력(이해) · 이성(조합/연산) · 출력(생성)을 분리.
- **효율이 우위 축.** ppl 천장은 transformer가 낮을 수 있으나, sparse+국소 학습은 **같은 계산당 성능
  (학습 효율)·저활성률·online 학습·해석 가능성**에서 이길 여지가 있다.

---

## 2. 목표 아키텍처와 현재 scaffold

### 목표 아키텍처 — 원래 방향 복귀

```text
하나의 sparse connectome W

입력 유닛 I  →  저장·이성 유닛 R  →  출력 유닛 O
                  ↕ 재귀 연결

hot  = 현재 활성 assembly·계산
warm = 빠르게 변하는 연결
cold = 공고화된 장기 연결·안정된 assembly
```

- I/R/O는 서로 다른 모델이나 알고리즘 행렬이 아니라 **한 연결망 안의 유닛 구역**이다. 각 유닛의 연결
  벡터는 전체 connectome의 해당 행이다.
- v12가 반증한 것은 이 원형 전체가 아니라, 어휘 전체를 R의 고정점 attractor로 저장한 단순 구현이다.
  기억 복원에는 안정상태를 쓸 수 있지만 문맥·순서·사고는 transient assembly trajectory도 사용한다.
- 선택적으로 각 유닛에 발달 중 움직일 수 있는 3차원 위치를 둔다. 유클리드 거리는 topology 형성에,
  assembly cosine은 단어·관념의 표상 유사도 측정에 사용한다. 우선 거리 기반 연결 형성만 검증하고 전파
  지연은 별도 가설로 보류한다.
- 첫 복귀 프로토타입은 `spatial_connectome.py`, 구조·학습 probe는 `test_spatial_connectome.py`다.

현재 프로토타입이 검증한 범위:

- 하나의 edge-list가 I/R/O의 모든 연결을 담고, 각 유닛 벡터가 그 그래프의 한 행으로 복원됨
- 거리 기반 연결의 평균 길이가 동일 해부학적 후보쌍 평균보다 짧음
- `A→B`와 `B→A`, 같은 마지막 토큰의 다른 앞 문맥이 서로 다른 hot 상태를 만듦
- 초기 위치가 활동에 따라 움직이고 위치 가소성이 감소함
- 국소 free/target phase 학습이 목표 출력 assembly의 활성을 높이고 warm 변화가 cold로 전달됨
- **문맥 분기 goal 통과** (`context_branch_probe.py`, 20 seeds): 동일한 마지막 입력 B에 대해
  `A,B→C`와 `D,B→E`를 동시에 학습했다. 성공은 두 출력 모두 top-1이고 두 R 상태 cosine<0.95인 경우로,
  각 topology에서 80% 이상을 요구했다.

| topology | 성공 | 평균 R-state cosine | 평균 연결 거리 |
|---|---:|---:|---:|
| random sparse | 19/20 | 0.620 | 0.628 |
| distance-biased | 18/20 | 0.352 | 0.505 |
| distance + position development | 20/20 | 0.445 | 0.493 |

첫 구현(token당 5 내부 step)은 마지막 B가 앞 문맥을 지워 R cosine 0.92~0.97, 성공 0/5였다. 새로운
기전을 더하지 않고 내부 step을 2로 줄여 transient를 보존하자 분기가 재현됐다. **거리 자체가 문맥 분기의
필수조건은 아니며**, 배선을 단축하는 보조 prior다. 현재 signed-LTD core에서는 위치 발달이 distance-only의
문맥 분기 성공률을 높이지 않았다.

**범주 일반화 goal 통과** (`category_generalization_probe.py`, 20 seeds): cat/dog/horse와 car/bus/van의
속성 경험으로 두 prototype을 만들고, 범주 target을 한 번도 받지 않은 wolf/fox/truck/bike를 R-assembly
cosine으로 분류했다. 각 held-out은 공통 속성 2개와 새 속성 1개를 가진다.

| topology | 완벽 seed | held-out | 무학습 | labeled-only | 속성교환 | R cosine gap |
|---|---:|---:|---:|---:|---:|---:|
| random sparse | 20/20 | **80/80** | 41/80 | 43/80 | **80/80** | +0.320 |
| distance-biased | 20/20 | **80/80** | 42/80 | 41/80 | **80/80** | +0.328 |
| distance + position development | 20/20 | **80/80** | 42/80 | 42/80 | **80/80** | +0.302 |

`labeled-only`는 prototype 개체만 속성학습하고 held-out은 무작위로 둔 통제다. `속성교환`은 wolf 같은 이름은
유지하면서 vehicle 속성을 주었을 때 예측이 vehicle로 뒤집히는지를 센다. 따라서 결과는 이름 우연이나
prototype 조직만으로 설명되지 않는다. 돌파구는 (1) property target과 entity free 상태 차이를 쓰는 국소
I→R delta(LTP+LTD), (2) 공동활성 R 유닛으로 입력당 4개 시냅스를 만들되 총 out-degree는 유지하는 구조
가소성, (3) 감각 seed를 최소사용 I/O 유닛에 균형 배치해 우연한 충돌과 타 단어 경로 덮어쓰기를 막은
것이었다. 이 마지막 변경 뒤 범주 회귀는 세 topology 모두 80/80으로 안정화됐다.

**범주 장기기억·언어출력 goal 통과** (`category_memory_output_probe.py`, 20 seeds). 사용자 goal 원문:

> "범주 표상을 장기 기억과 언어 출력에 연결"

animal/vehicle 출력 target은 cat/dog/horse와 car/bus/van에만 주고, wolf/fox/truck/bike에는 끝까지
범주 target을 주지 않았다. 활성된 흥분성 R 유닛마다 목표 O assembly로 가장 약한 R→O 시냅스 하나를
교체하고, 한 번 더 전파한 O의 실제 활성에서 `pre_R × (teacher_O - post_O)` 국소 delta를 학습했다. 새
시냅스는 warm에서 시작하며 총 out-degree는 변하지 않는다. 이후 100회 공고화하고 warm·hot을 0으로 지운
뒤, cosine prototype head 없이 O assembly 합의 top-1을 실제 출력 토큰으로 읽었다.

| topology | 완벽 seed | cold-only | 출력 미학습 | 공고화 전 warm 제거 | labeled-only | 속성교환 | 후속학습 뒤 보존 | 새 범주 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random sparse | 16/20 | **80/80** | 42/80 | 45/80 | 43/80 | **80/80** | **77/80** | **39/40** |
| distance-biased | 10/20 | **79/80** | 38/80 | 36/80 | 46/80 | **79/80** | **68/80** | **38/40** |
| distance + position development | 10/20 | **80/80** | 36/80 | 37/80 | 40/80 | **78/80** | **69/80** | **37/40** |

완벽 seed는 cold 4개, 후속학습 뒤 기존 4개, 새 held-out 2개를 모두 맞힌 경우다. 두 번째 세션에서는
fruit/tool prototype만 출력 label을 받고 peach/drill은 속성만 학습했다. 기존 항목도 animal/vehicle만이
아니라 네 출력 단어 전체와 경쟁하므로 닫힌 옛 decoder를 재사용한 결과가 아니다. 공고화 전 warm 제거는
chance 부근이지만 공고화 후 cold-only는 98.8~100%였고, 같은 이름의 속성을 반대로 주면 출력 단어도
97.5~100% 뒤집혔다. 따라서 저장된 것은 이름별 category 답표가 아니라 범주 R geometry에서 O 언어
assembly로 가는 연결이다.

후속학습 뒤 보존은 85.0~96.3%로 verifier 하한은 통과했지만 완전 무망각은 아니다. 이 단계의 출력은 주어진
범주 어휘 중 한 토큰을 고르는 O-assembly readout이었고, 질의 문맥과 전체 어휘 경쟁은 아래 다음 goal에서
따로 검증했다. 여러 토큰 자유 문장 생성, 계층·다중라벨 범주는 아직 증명하지 않았다. 출력 반복을 20회보다
늘리면 포화와 함께 간섭이 커졌고, O를 64→128로 늘린 ablation도
성능이 악화되어 둘 다 기각했다. O 확장이 고정 out-degree에서 R 재귀 연결의 상대적 몫을 줄였을 가능성은
있지만 이번 probe에서 원인 자체를 분리 측정하지는 않았다.

**두 트랙 end-to-end 질의응답 goal 통과** (`dialogue_qa_probe.py`, 10 seeds). 학습 데이터의 생성·제공은
환경 상호작용을 흉내 낼 수 없으므로 생물 국소성의 필수조건에서 제외했다. 두 트랙 모두 동일한 합성 사실
record와 prototype 답 supervision을 받는다. wolf/fox/truck/bike와 이후 peach/drill은 속성 사실만 받고
질의 정답 target은 한 번도 받지 않는다. 같은 개체에 `what_is wolf→animal`과
`what_feature wolf→fur`를 물어 query binding을 검사하며, 추론 API에는 후보 답 목록을 넘기지 않고 등록된
모든 O-token이 경쟁한다.

- **기능 본선** (`FunctionalAssemblyDialogue`): token을 16/512 sparse unit assembly로 유지하고, 전역
  associative semantic/output memory로 기능 상한을 측정한다.
- **생물 국소 본선** (`BioLocalAssemblyDialogue`): 기존 단일 connectome에서 query assembly가 상보적인
  R dendritic subcircuit를 열고, local `pre_R × (teacher_O-post_O)`와 source-local stochastic
  synaptogenesis/pruning을 사용한다. warm→cold 뒤 fast state를 지우며 후속학습에는 6 round마다 old
  prototype을 replay한다.

| track/topology | 완벽 seed | 첫 held-out 질의 | gate 제거 | 답 미학습 | warm 제거 | no-rewire | no-replay 보존 | replay 보존 | 새 held-out 질의 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| functional/global | **10/10** | **80/80** | 40/80 | 0/80 | — | — | — | **80/80** | **40/40** |
| bio/random (본선) | **5/10** | **77/80** | 40/80 | 5/80 | 3/80 | 76/80 | 70/80 | **74/80** | **37/40** |
| bio/distance | 2/10 | 78/80 | 38/80 | 2/80 | 4/80 | 71/80 | 56/80 | 67/80 | 36/40 |
| bio/developed | 0/10 | 76/80 | 40/80 | 0/80 | 2/80 | 71/80 | 38/80 | 61/80 | 35/40 |

기능 본선은 전 항목 100%다. 생물 본선으로 선택한 random sparse 조건도 첫 질의 96.3%, cold/replay 후
보존 92.5%, 새 질의 92.5%로 verifier를 통과했다. gate 제거가 정확히 50%로 떨어져 질문별 dendritic
routing이 같은 entity의 두 답을 분리한 핵심이고, warm 제거 3.8%는 cold 공고화 필요성을 보인다. replay는
보존을 87.5→92.5%로 높였다. no-rewire가 이미 95%여서 stochastic structural plasticity의 초기 추가 이득은
1/80에 불과했다. 물리 거리와 위치발달은 이번 후속 보존을 악화시켜 본선으로 채택하지 않았다.

**locality audit:** sparse assembly, edge-local propagation, local synaptic value update, local dendritic gate
적용, source-local stochastic pruning/formation, no autograd/weight transport는 충족한다. 남은 전역 scaffold는
영역 top-k/max activity control, seed/query-gate 균형 배정, supervised target O clamp, 전체 O vocabulary
argmax다. 데이터 supervision 자체는 허용된 경계조건이지만, target signal의 내부 전달과 나머지 세 장치는
완전한 생물 국소 구현으로 세지 않는다. 현재 대화 행동은 구조화된 두 query intent에 한 O-token으로 답하는
단계이며, 자연어 문장 parser와 여러 token 응답 생성은 다음 범위다.

```bash
python3 -m unittest -v test_spatial_connectome.py
python3 spatial_connectome.py
python3 context_branch_probe.py --seeds 20 --rounds 180 --verify
python3 category_generalization_probe.py --seeds 20 --verify --quiet
python3 category_memory_output_probe.py --seeds 20 --verify --quiet
python3 dialogue_qa_probe.py --seeds 10 --verify --quiet
```

### 현재 성능 scaffold

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

이 scaffold의 `Embedding`, `P1/P2`, `W_bi/W_co`, Q/K/V와 별도 head는 메커니즘별 성능을 진단하기 위해
분해한 장치다. 원래의 단일 connectome 구현과 같지 않으며, 실험 결과를 보존한 채 목표 구조로 통합하는
과정이 다음 본선이다.

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
| — | softmax 대체 탐색(`multiplicative_gate.py`): 승산게이트·비교게이트·행렬혼합·divisive norm·측면억제(dt포함) | 정규화 계열(gate/cmpgate/matmul/divnorm/lateral) 전부 **59.4~62.7 (천장 확정)** — 전부 attn 57.4엔 못 미침 |
| — | phase(binding-by-synchrony, 정규화 아닌 위상결맞음 게이팅) | beta=2.0 **58.4 최고** — 정규화 계열 천장(59.4) 넘음, attn(57.4)과 격차 2p→1p로 좁힘 |
| — | phase2(Kuramoto 상호결맞음, window 전체 population coupling) | it=1~8 전부 58.8~59.0 — **개선 없음**, phase 단독(58.4)보다 오히려 약간 나쁨 |
| — | order_probe.py: window 순서 민감도 구조 검증 | **attn/divnorm/lateral/phase/phase2 전부 순서-blind**(memory 슬롯 뒤바꿔도 출력 완전동일, Δ=0) — add/gate/cmpgate/matmul만 순서 봄. attn 계열 우위는 "순서 활용"이 아니라 "content 기반 경쟁적 선택"이었음이 드러남 |
| — | phase_pos(theta-gamma 순서코드: 고정 위치위상 + content위상 결합) | 구조적으론 order-sensitive(Δ=0.58) 확인됨 — 근데 beta=1.0/2.0/4.0 **59.7/60.3/61.1로 전부 phase 단독(58.4)보다 나쁨**. 순서정보 추가가 이 벤치마크(K=8 vocab2000, bigram위주)엔 오히려 해가 됨 |

---

## 4. 핵심 발견 (진단 사슬)

1. **붕괴 2종 분리.** ①부호붕괴(all-negative): predictive-coding target이 mostly-0이라 오차가 전부
   음수 → vocab 무관. **err-centering**(오차 zero-mean; cross-entropy 기울기에 내장)으로 해결. ②attractor
   붕괴(단일 고정점): 고전 Hopfield식 구현으로 재귀망에 vocab 전체를 고정점으로 저장하면서 용량을 초과했다.
   **이는 단일 connectome 전체의 반증이 아니라 token=fixed-point 가정의 실패다.**
2. **표현학습은 bio-국소로 가능.** feedback alignment(backprop 없음)가 임베딩을 학습하고 붕괴하지 않는다
   (vocab2000: FA 83→ 하락, backprop 62.9). 콜드-웜-핫 scaffold(구조 prior + FA 학습)가 무작위 init보다
   우수. → "credit assignment = 하드월"은 반증됨. 남은 건 성능 격차.
3. **구조화 피드백 > 무작위.** hidden 피드백: random-FA 70 < sign-concordant 64.5 < backprop 60.4.
   부호 일치(E/I 일관) 피드백이 격차 대부분을 메운다. 순수 무작위는 하한 증명이고, 실제 뇌는 구조화
   피드백(predictive coding 방향)을 쓸 것.
4. **조합(binding)이 남은 레버.** 덧셈 결합은 순서/상호작용을 못 잡아 천장(58). softmax 라우팅
   (attention)이 additive를 이긴다(58.6 vs 61.8). 다만 pairwise Hadamard(곱을 feature로)는 실패
   (product를 *score*로 써야). 완전 폐쇄(→53)는 transformer 스택 필요 = 비-bio.
5. **간단한 대화의 형식만 달성.** 큰 vocab(→ `<unk>` 홍수 제거) + 대화 데이터(empathetic dialogues,
   턴마커)로 frozen-routing 모델이 공감 레지스터로 응답한다. 그러나 사실·추론·구체적 지시 probe에서
   실패해 제네릭 공감기로 판정했다. 완전-bio(DFA) 버전도 형식은 유지하지만 품질 비용이 있다.
6. **softmax 없는 window 경쟁도 격차를 완전히 못 메운다.** 비가환 결합(승산게이트·비교게이트·행렬혼합,
   60.2~60.5)은 순서보존은 되나 attn과 격차 그대로 — 원인은 비가환성이 아니라 window 전체 경쟁 정규화
   부재. divisive normalization(Carandini & Heeger, 실측 지수 n≈2)이 최고 non-softmax(59.4)이나 여전히
   attn(57.4)에 ~2p 못 미침. **정확한 뇌 정규화 공식도 softmax의 계산력에 못 미친다는 실측 증거** — 값
   조정(n=1/4)도 실측치(n=2)를 못 이김. 반복적 측면억제(정적 공식 대신 여러 스텝 수렴)가 남은 후보.

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
- **공식 이식 2분류.** (a) 형태 유지 + 상수만 조정(예: divnorm n-sweep — 실측 뇌 값 n=2가 조정값보다
  우수) vs (b) 형태 버리고 뜻만 단순 재현(예: v11 log-Hebbian). 어느 쪽도 항상 우월하지 않음 — 매번 실측
  필요. top-down 재구성이되 뇌의 정확한 공식을 통째로 이식하진 않는다(GCI의 bottom-up·물리원시 원칙과
  구분).
- **공식의 적용 스케일 확인.** 뇌 전반(피질 국소회로) 겨냥 공식을 잘못된 pool 범위에 적용하면 형태가
  맞아도 붕괴할 수 있다(v9 divisive-readout이 국소 pool 대신 전체 N유닛 평균으로 나눈 사례). 이식 전
  "이 공식이 원래 요구하는 pool/스케일이 뭔지" 확인할 것.
- **공식레벨 vs 구조레벨 bio 근거 구분.** 위 항목들은 단일 연산의 수식(공식레벨). 이와 별개로 입력/이성/
  출력 3분할(Fedorenko, 언어망 vs 추론망)과 콜드-웜-핫 다중시간척도(CLS·Benna-Fusi)는 구조레벨 bio 근거이며
  독립적으로 유효 — 한쪽(예: softmax 대체) 탐색을 접어도 다른쪽엔 영향 없음.

---

## 7. 한계 · 열린 문제

- **bio-국소 attention.** 라우팅 이점을 고정 무작위(reservoir)로 잡았으나 softmax-dot은 완전 bio가 아님.
  비가환 결합·divisive norm(실측지수 포함, 최고 59.4)·측면억제 다 시도했으나 attn(57.4)과 ~2p 격차 지속 —
  정규화 계열은 천장에 근접한 것으로 보임. 뇌의 binding(theta-gamma 위상부호, 수상돌기 동시검출)처럼
  정규화가 아닌 다른 축은 아직 미시도.
  **다음 갈래 4개(2026-07-08 정리)**:
  1. 동역학 부활 — 정적 공식 대신 반복수렴(재귀)으로, 이때 국소 pool(단일/군집 뉴런 스케일) vs 뇌 전역
     식 스케일 구분 후 적용 (design-principles ⑧ 스케일 미스매치 원칙과 연결).
  2. 공식의 뜻만 유지한 채 단순화해 적용 (⑦-b 계열, 예: v11 log-Hebbian 방식).
  3. softmax를 non-bio 예외로 그냥 수용 — 정규화 계열 천장(~59) 확인되면 이 프로젝트 본선(대화·효율축)
     으로 복귀.
  4. 위상코드(theta-gamma) — 지금까지(divnorm/lateral/gate) 전부 "경쟁 정규화" 계열이었음. 진동 위상으로
     슬롯 나눠 순서를 인코딩하는, 정규화가 아닌 완전히 다른 축. 결과 예측 불가, 코드 처음부터 새로 작성 필요.

  주의: 1과 2는 배타적 갈림길이 아님 — divnorm(n=2, 59.4)이 이미 둘의 동시적용(K=8 국소pool로 스케일
  맞춤 + relu·거듭제곱으로 원식 단순화)이었고, lateral+dt 실험도 1을 반복수렴까지 확장한 버전.

  **판정 완료(dt 스윕, n=2/beta=0.3 고정, 오일러 리키적분)**: dt=0.3→59.8(최선) | dt=0.5→62.1 |
  dt=0.7→61.6 | dt=1.0→60.9. **전부 divnorm 59.4 못 넘음.** 1+2(동역학 갈래) 여기서 막힘 — 정규화
  계열(gate/cmpgate/matmul/divnorm/lateral 전부) 천장 = **~59.4로 최종 확정**.

  **4(위상코드) 실행 결과 — 돌파.** `phase` 모드 구현: content-match score를 위상 φ∈[0,π/2]로 변환
  (β=steepness, 높은 매치→φ→0 in-phase), coherence gain w=cos(φ)² (Malus's law)로 게이팅 — **window
  전체 합으로 정규화하지 않음**(softmax/divnorm/lateral 전부가 공유하던 "경쟁적 정규화" 성질 자체를
  뺀 것, Fries 2005 communication-through-coherence 유비). β 스윕: 0.5→58.5 | 1.0→58.9 | **2.0→58.4
  (최고)** | 4.0→62.0(너무 가파름, 이진게이트화되며 붕괴). **정규화 계열 천장(59.4)을 처음으로 넘음
  — attn(57.4)과 격차 2p→1p로 좁힘, 지금까지 non-attn 최고 기록.** softmax의 핵심이 "정규화"가
  아니라 "매치 기반 결맞음(비정규화)" 자체였을 가능성 시사.

  **phase2(Kuramoto 상호결맞음) 결과 — 개선 없음.** window 토큰끼리 위상을 서로 끌어당기는 진짜
  population coupling(`coup_ij=tanh(k_i·k_j/√d)` + Kuramoto 갱신, iters 스윕) 구현·실행. it∈{1,3,5,8}
  전부 **58.8~59.0에 정체** — phase 단독(58.4)보다 오히려 약간 나쁨. coupling 추가가 도움 안 됨.

  **구조 검증(`order_probe.py`, 신규) — 더 큰 발견.** 훈련 없이 순수 구조로: window의 memory 슬롯
  (위치 1..K-1)을 뒤바꿔도 출력이 바뀌는지 확인. **attn(진짜 softmax attention 포함)·divnorm·
  lateral·phase·phase2 전부 순서-blind**(Δlogit=0.000000, 부동소수점까지 완전동일) — 이 코드의
  attn류엔 애초에 positional encoding이 없어서 mix=Σw_j·v_j가 j에 대한 대칭합이 되기 때문(transformer가
  별도 위치임베딩을 반드시 쓰는 이유의 실측 재현). add/gate/cmpgate/matmul(슬롯별 가중치·재귀)만 순서를
  봄. **재해석: attn 계열의 우위(divnorm 대비도, attn의 gate 대비 우위도)는 "순서를 더 잘 쓴다"가
  아니라 "content 기반 경쟁적 선택 방식의 차이"였음** — phase2가 겨냥한 "결맞음=진짜 binding" 전제도
  이 관점에서 재검토 필요(순서정보 자체가 안 들어가므로). **새 레버 후보**: k_j에 위치신호(고정
  sinusoidal 등, 유전 prior 원칙과 부합) 추가 — 지금까지 어떤 attn류도 안 써본 축이라 attn(57.4)·
  phase(58.4) 둘 다 더 낮아질 여지 있음, 미시도.

  **레버 시도 결과 — phase_pos, 실패.** theta-gamma serial-order 코드(Lisman & Idiart 1995): 슬롯
  위치 j마다 고정 위상 오프셋 `pos_j=(π/2)·j/(K-1)`(j=0="지금"~j=K-1=가장 오래됨)을 content-match
  위상에 더해 "무엇"과 "언제"를 한 게이트에 결합. order_probe로 구조적 order-sensitivity 확인(Δ=0.58,
  phase/phase2는 0). **그러나 학습 결과: β=1.0→59.7 | β=2.0→60.3 | β=4.0→61.1, 전부 phase 단독
  (58.4)보다 나쁨.** 고정 선형 위치감쇠가 이 벤치마크(K=8, vocab2000, bigram 위주)엔 유용 신호보다
  노이즈로 작용한 것으로 보임 — 순서정보 자체는 넣었지만 감쇠 형태/스케일이 안 맞을 가능성, 또는
  이 태스크가 애초에 순서보다 국소통계 의존이 커서(arc 반복 확인 사실) 순서 신호의 실익이 작을 가능성
  둘 다 남음, 구분 안 됨. **현재 세션 최종 결론: phase(β=2.0, 58.4)가 이 arc의 non-attn 최고 기록으로
  유지**, phase2·phase_pos 둘 다 개선 실패.

  **별도 축(구조레벨, design-principles ⑨) — 위 4갈래와 독립, 접어도 서로 안 막힘**: 입력/이성/출력
  3분할(이미 구현, Fedorenko 언어망-추론망 분리 근거)과 콜드-웜-핫 다중시간척도(부분구현, 웜→콜드 공고화
  단계 미구현)도 뇌 기전 근거를 가진 별도 고려 대상 — softmax 대체는 공식레벨 실험이라 이쪽엔 영향 없음.
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
