# pattern-is-all-you-need

**The model is a network of vectors. A vector *is* its connection weights to the other vectors, not a row
in a table those weights are applied to. A token or a concept is the set of vectors that activate
together.** That is where this started, and everything else follows from it.

From there I borrow mechanisms from biology, because the brain is the one system that already computes
this way. The end goal is a model that uses this representation in real dialogue. **Simple dialogue** is
not the final definition of success; it is the first external milestone.

Sparsity is not the objective. Assemblies come out sparse here because a sparse prior is given, and the
guess — and it is a guess, and partly a rationalization — is that a network of this kind gets sparse on
its own at brain-like scale. Nothing in this repository tests that.

> The full process, evidence, and the paths I rejected live in my wiki at
> `research/pattern-is-all-you-need.md` (#1–#33). This README is a summary of the current state and how
> to reproduce it.

### Current direction

Two tracks, differing in exactly one thing — whether locality is required:

1. **Functional track** — *assembly representation + dialogue.* Tokens and concepts stay vector/unit
   assemblies, and the model has to be usable in real dialogue. Any engineering learning method is
   allowed; the obligation is to record separately which non-local device contributed which part of the
   performance. This track measures the ceiling and finds the bottleneck.
2. **Bio-local track** — *assembly representation + local biological mechanism + dialogue.* The same
   dialogue ability, built from local synaptic information and local dynamics. Global error, global
   top-k, global weakest-edge search and global seed balancing do not count as biological success; they
   get replaced by local mechanisms one at a time. This track is the main line.

Both share the same invariant and the same destination: **encode tokens and concepts as vector/unit
assemblies rather than single embedding rows, and make context, category, memory, and thought operate
out of those connections and activations, surfacing as dialogue.** A functional-track result is never
counted as bio-track evidence.

**Breakthrough principle**: when the bio track gets stuck, I do not smuggle attention/backprop back in.
I form computational hypotheses from biological mechanisms — local inhibition, dendritic feedback,
structural plasticity, replay, consolidation, neuromodulation — and test them with functional and
ablation experiments. What gets transplanted is the shape of the problem the brain solves, not the
brain's exact formulas; the constants are mine to fit.

**Evaluation principle**: dialogue quality on the functional track, and locality plus dialogue quality on
the bio track, are measured in separate tables.

### Layers of the goal

1. **Substrate**: input, storage/reasoning, and output units form one sparse connectome, and tokens and
   concepts are represented as unit assemblies.
2. **Internal capability**: context-dependent state trajectories, category sharing, long-term memory
   consolidation, order, composition, and thought — each probed directly.
3. **Linguistic behavior**: use those capabilities to generate dialogue that matches the content.
   Producing only fluent empathetic sentences does not count as success.

---

## 1. Thesis

- **A vector is its own weights.** Weights are not a separate matrix applied to vectors; a unit's vector
  *is* its outgoing row of the connectome. This is the starting point, not a compression trick.
- **Token/concept = cell assembly.** Sensory input may hand over a stable seed, but the assemblies and
  semantic relations in the storage/reasoning substrate have to form out of connectivity and activity.
  Assemblies are sparse here because a sparse prior is given, not because sparsity is the aim.
- **Two learning tracks.** The functional track keeps the assembly representation and looks for the
  ceiling and the bottleneck of dialogue capability. The bio track reproduces the same ability with local
  rules and no weight transport; feedback alignment is not the final answer either, it is an engineering
  scaffold to compare against.
- **Language ≠ thought.** The language network (comprehension/generation I/O) and the reasoning network
  (multiple-demand) are separate (Fedorenko et al.) → split into input (comprehension), reasoning
  (composition/computation), and output (generation).
- **Efficiency is the axis where this can win.** A transformer may reach a lower perplexity ceiling, but
  sparse + local learning has room to win on **performance per unit of compute (learning efficiency), low
  activation rate, online learning, and interpretability**.

---

## 2. Target architecture and the current scaffold

### Target architecture — return to the original direction

```text
one sparse connectome W

input units I  →  storage/reasoning units R  →  output units O
                    ↕ recurrent connections

hot  = currently active assembly / computation
warm = fast-changing connections
cold = consolidated long-term connections, stabilized assemblies
```

- I/R/O are not separate models or algorithmic matrices but **regions of units inside one connectome**.
  Each unit's connection vector is the corresponding row of the whole connectome.
- What v12 refuted is not this whole blueprint but one simple implementation of it: storing the entire
  vocabulary as fixed-point attractors in R. Memory recall may use stable states, but context, order, and
  thought also use transient assembly trajectories.
- Optionally each unit gets a 3D position that can move during development. Euclidean distance is used for
  topology formation; assembly cosine is used to measure representational similarity of words and concepts.
  I validate distance-based connection formation first and defer propagation delay as a separate hypothesis.
- The first prototype of the return is `spatial_connectome.py`; the structure/learning probes are in
  `test_spatial_connectome.py`.

What the current prototype has validated:

- One edge list holds every I/R/O connection, and each unit vector is recoverable as a row of that graph
- Distance-based connections have a shorter mean length than the mean over the same anatomical candidate pairs
- `A→B` vs `B→A`, and different preceding contexts with the same last token, produce different hot states
- Initial positions move according to activity, and positional plasticity decays
- Local free/target-phase learning raises the activation of the target output assembly, and warm changes
  transfer into cold
- **Context-branching goal passed** (`context_branch_probe.py`, 20 seeds): with the same last input B, I
  trained `A,B→C` and `D,B→E` at once. Success requires both outputs top-1 and the two R-state cosines
  <0.95, at ≥80% per topology.

| topology | success | mean R-state cosine | mean connection length |
|---|---:|---:|---:|
| random sparse | 19/20 | 0.620 | 0.628 |
| distance-biased | 18/20 | 0.352 | 0.505 |
| distance + position development | 20/20 | 0.445 | 0.493 |

The first implementation (5 internal steps per token) let the final B erase the preceding context: R cosine
0.92–0.97, success 0/5. Cutting internal steps to 2 to preserve the transient brought branching back, with no
new mechanism added. **Distance itself is not necessary for context branching**; it is an auxiliary prior that
shortens wiring. In the current signed-LTD core, position development did not raise distance-only's
context-branching success rate.

**Category generalization goal passed** (`category_generalization_probe.py`, 20 seeds): two prototypes were
built from property experience with cat/dog/horse and car/bus/van, and wolf/fox/truck/bike — which never
received a category target — were classified by R-assembly cosine. Each held-out item has 2 shared properties
and 1 new one.

| topology | perfect seeds | held-out | no learning | labeled-only | property swap | R cosine gap |
|---|---:|---:|---:|---:|---:|---:|
| random sparse | 20/20 | **80/80** | 41/80 | 43/80 | **80/80** | +0.320 |
| distance-biased | 20/20 | **80/80** | 42/80 | 41/80 | **80/80** | +0.328 |
| distance + position development | 20/20 | **80/80** | 42/80 | 42/80 | **80/80** | +0.302 |

`labeled-only` is a control where only the prototype entities get property learning and the held-outs are left
random. `property swap` counts whether the prediction flips to vehicle when a name like wolf is kept but given
vehicle properties. So the result is not explained by name coincidence or by prototype organization alone. The
breakthrough was (1) a local I→R delta (LTP+LTD) using the difference between the property-target and
entity-free states, (2) structural plasticity that creates 4 synapses per input onto co-active R units while
holding total out-degree fixed, and (3) balancing sensory seeds across the least-used I/O units to prevent
accidental collisions and the overwriting of other words' pathways. After that last change, category
regression stabilized at 80/80 across all three topologies.

**Category long-term memory + language output goal passed** (`category_memory_output_probe.py`, 20 seeds).
Goal: connect category representations to long-term memory and language output.

animal/vehicle output targets went only to cat/dog/horse and car/bus/van; wolf/fox/truck/bike never received a
category target at any point. For each active excitatory R unit, the single weakest R→O synapse is replaced by
one toward the target O assembly, and the local delta `pre_R × (teacher_O - post_O)` is learned from O's actual
activation after one more propagation. New synapses start warm and total out-degree does not change. After 100
consolidation rounds, warm and hot are zeroed and the top-1 of the summed O assembly is read out as the actual
output token — no cosine prototype head.

| topology | perfect seeds | cold-only | output unlearned | warm removed before consolidation | labeled-only | property swap | retention after later learning | new category |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random sparse | 16/20 | **80/80** | 42/80 | 45/80 | 43/80 | **80/80** | **77/80** | **39/40** |
| distance-biased | 10/20 | **79/80** | 38/80 | 36/80 | 46/80 | **79/80** | **68/80** | **38/40** |
| distance + position development | 10/20 | **80/80** | 36/80 | 37/80 | 40/80 | **78/80** | **69/80** | **37/40** |

A perfect seed gets all 4 cold items, all 4 prior items after later learning, and both new held-outs right. In
the second session only the fruit/tool prototypes received output labels; peach/drill learned properties only.
Prior items compete against all four output words, not just animal/vehicle, so this is not the result of reusing
a closed old decoder. Removing warm before consolidation lands near chance, but cold-only after consolidation is
98.8–100%, and giving the same name the opposite properties flips the output word 97.5–100% of the time. What is
stored is therefore not a per-name category answer table but a connection from category R geometry to O language
assemblies.

Retention after later learning is 85.0–96.3% — it clears the verifier's lower bound but is not complete absence
of forgetting. The output at this stage was an O-assembly readout choosing one token from the given category
vocabulary; query context and full-vocabulary competition are validated separately in the next goal below.
Multi-token free sentence generation and hierarchical/multi-label categories are still unproven. Pushing output
repetitions past 20 increased saturation and interference, and an ablation raising O from 64→128 also degraded
performance, so both were rejected. It is possible that expanding O reduced the relative share of R's recurrent
connections under fixed out-degree, but this probe did not isolate that cause by measurement.

**Two-track end-to-end question answering goal passed** (`dialogue_qa_probe.py`, 10 seeds). Generating and
handing over training data cannot imitate environmental interaction, so it is excluded from the necessary
conditions for biological locality. Both tracks receive the same synthetic fact records and prototype answer
supervision. wolf/fox/truck/bike, and later peach/drill, receive property facts only and never a query answer
target. Query binding is tested by asking the same entity both `what_is wolf→animal` and
`what_feature wolf→fur`, and the inference API is not handed a candidate answer list — every registered O-token
competes.

- **Functional track** (`FunctionalAssemblyDialogue`): keeps tokens as 16/512 sparse unit assemblies and
  measures the functional ceiling with global associative semantic/output memory.
- **Bio-local track** (`BioLocalAssemblyDialogue`): inside the existing single connectome, the query assembly
  opens a complementary R dendritic subcircuit, and uses local `pre_R × (teacher_O-post_O)` plus source-local
  stochastic synaptogenesis/pruning. Fast state is wiped after warm→cold, and later learning replays old
  prototypes every 6 rounds.

| track/topology | perfect seeds | first held-out query | gate removed | answer unlearned | warm removed | no-rewire | no-replay retention | replay retention | new held-out query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| functional/global | **10/10** | **80/80** | 40/80 | 0/80 | — | — | — | **80/80** | **40/40** |
| bio/random (main) | **5/10** | **77/80** | 40/80 | 5/80 | 3/80 | 76/80 | 70/80 | **74/80** | **37/40** |
| bio/distance | 2/10 | 78/80 | 38/80 | 2/80 | 4/80 | 71/80 | 56/80 | 67/80 | 36/40 |
| bio/developed | 0/10 | 76/80 | 40/80 | 0/80 | 2/80 | 71/80 | 38/80 | 61/80 | 35/40 |

The functional track is 100% on every item. The random sparse condition chosen as the bio main line also cleared
the verifier: first query 96.3%, retention after cold/replay 92.5%, new queries 92.5%. Gate removal drops to
exactly 50%, so per-question dendritic routing is the core of separating one entity's two answers, and warm
removal at 3.8% shows cold consolidation is necessary. Replay raised retention from 87.5→92.5%. no-rewire was
already at 95%, so stochastic structural plasticity added only 1/80 initially. Physical distance and position
development made this retention worse, so they were not adopted for the main line.

**Locality audit:** sparse assemblies, edge-local propagation, local synaptic value update, local dendritic gate
application, source-local stochastic pruning/formation, and no autograd/weight transport are all satisfied. The
remaining global scaffolds are region-wide top-k/max activity control, balanced seed/query-gate allocation,
supervised target O clamp, and full O-vocabulary argmax. Data supervision itself is an allowed boundary
condition, but the internal delivery of the target signal and the other three devices do not count as a complete
bio-local implementation. The dialogue behavior at this point is answering two structured query intents with a
single O-token; a natural-language sentence parser and multi-token response generation are the next scope.

**Boundary-free syllable→chunk→QA goal passed** (`syllable_chunk_probe.py`, 10 seeds). External preprocessing
provides only Unicode normalization and a fixed Hangul syllable repertoire. The input is
`늑대가속한종류를말해 → [늑, 대, 가, 속, 한, 종, 류, 를, 말, 해]` with whitespace and word/eojeol boundaries
stripped, and no entity ID, query intent, or answer candidate list is passed to the API. No evaluation question
is identical to a training sentence containing `종류`/`특징`.

- **Functional track** (`FunctionalSyllableDialogue`): builds 2–6 syllable chunk assemblies from repeated
  substring statistics and uses global semantic/conjunctive output memory — this is the ceiling.
- **Bio-local track** (`ConnectomeSyllableDialogue`): strengthens only edges with active pre/post between
  adjacent syllable assemblies, and repeated paths crossing a fixed local threshold recruit a sparse chunk
  assembly. When three boundary-free fact sentences are supplied as one supervised episode, the chunk repeated
  across all three becomes the entity assembly and feeds directly into the existing single-`SpatialConnectome`
  I→R concept learning. Query chunks form 6 control prototypes by novelty competition with no intent label, and
  open a 12.5% R dendritic subcircuit. From there it uses the existing local I→R concept rule and full
  O-vocabulary readout. Output memory uses a per-O-neuron dendritic coincidence branch instead of a single
  linear R→O sum. A new warm branch is recruited only when the target O clamp and current R activity do not fit
  an existing branch well enough, and after warm→cold a mature branch is not overwritten.

| track | fully-perfect seeds | first QA | retention after later learning | new QA | control removed | temporal removed | warm removed | linear R→O control | branch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| functional/global | **6/10** | **80/80** | **77/80** | **34/40** | 34/80 | 19/80 | — | — | — |
| bio/single-connectome | **6/10** | **78/80** | **72/80** | **36/40** | 40/80 | **0/80** | 20/80 | 56/80 + 34/40 | 214 |

The new stability–plasticity goal passed without replay. After consolidating the base animal/vehicle into cold,
only fruit/tool facts and answers were learned — no old record and no old answer target were given again. On the
bio track, retention and new learning were each 90%, and 6/10 seeds got every held-out answer from both sessions
right. With the same local value rule and data but dendritic branches off and a single linear R→O compartment,
it was 70% prior / 85% new. Branch separation raised total accuracy by 18/120, and raised retention in
particular by 20 points.

The mechanism has three parts. Only the contacts where the target clamp and the active pre coincide get a
stability tag, so the absence of non-answer LTD is not protected long-term as well. I→R concept edges past the
consolidation threshold are excluded from that same source's later pruning candidates. An O neuron compares only
the coincidence between its own existing branches and the current R pattern to recruit a novel branch, and when
capacity is full a cold mature branch is not replaced. Branches are stored inside `SpatialConnectome` as
per-O-neuron warm/cold state, and the whole O-token assembly competes as before.

**New locality audit:** the bio main line satisfies boundary-free syllable input, local temporal edge update,
local query prototype update, single-connectome I/R/O semantic/output memory, local concept/output synaptic
update, target-specific synaptic tag, O-local dendritic branch recruitment, mature-branch preservation, and no
autograd/weight transport. The remaining global scaffolds are finite-window enumeration, intersection of common
chunks across a fact episode, novelty winner and sparse activity competition, balanced chunk/control seed
recruitment, R-state somatic gain normalization, supervised property/O target clamp, and full-vocabulary argmax.
In particular, supervised episode grouping is a stand-in for the real-world situation of seeing the same object
alongside several utterances, and does not count as a biological locality success.

**Answer-sentence generation goal passed** (`syllable_sentence_probe.py`, 10 seeds). The answer is now an ordered
syllable emission rather than one O-token — `늑대가속한종류를말해` → `늑대는동물이야` — and held-out entities
still never receive a sentence target. There is no candidate list, no teacher forcing at inference, and no
external autoregressive decoder.

| track | exact | entity | content | frame | form-only | perfect seeds | no replay | no trace | no query gate | branches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| functional/global | **80/80** | 80/80 | 80/80 | 80/80 | 0/80 | **10/10** | — | — | — | — |
| bio/single-connectome | **76/80** | **80/80** | **76/80** | 77/80 | 1/80 | **7/10** | **0/80** | **0/80** | 40/80 | 251 |

Success is scored so that a fluent frame with wrong content cannot pass: the entity syllables, the content
syllables and the frame are counted separately, and sentences whose frame is right but whose content is wrong are
counted on their own line. That line reads 1/80, so the #19 generic-empathizer failure did not return. The four
remaining errors are category misclassifications (`늑대는탈것이야`), the known residual of the goals above, not
sentence failures.

Three local mechanisms carry it, each found by diagnosis rather than designed up front. A chunk assembly is linked
to the syllables it was recruited from; plain Hebbian let frequent syllables win on the units a chunk shares with
other chunks, so the LTP+LTD delta already used for concepts supplies the negative term. The name replay and the
semantic sequence are separate pathways, and the replayed name never enters the emission trace — a name is
open-class and cannot generalize from closed-class category prototypes, which is why removing the replay drops
entity accuracy to 0/80. A branch coincidence normalizes its meaning stream and its order stream apart and
multiplies them; concatenated into one normalized vector the two compete for the same budget, and raising the
emitted trace enough to end a sentence starves the query gate that selects the content.

The sentence also ends without a terminal symbol or a length limit. The order stream carries no entity
information: it reads 1.0 at every syllable the sentence still owes, 0.54 at the first spurious repeat and 0.14 at
the next, invariant across seeds and entities, while the full score over the same span swings between 0.61 and
2.38. So emission reuses the recruitment criterion — a branch either recognizes the present trace or the sentence
is over.

**Sentence locality audit:** satisfied — local chunk-constituent delta, name replay from the entity's own chunk,
separated repetition and semantic pathways, two-stream dendritic coincidence, order-gated termination,
single-connectome memory and output, no autograd/weight transport. Remaining — this goal *added two* global
scaffolds: the constituent delta sweeps every learned chunk, and membership is read by matching against whole
syllable assemblies. With the six inherited above that makes nine. The count has gone 4 → 7 → 9 across the last
three goals, so "remove the global scaffolds one at a time" has not been a net reduction: each new capability has
brought new global machinery. This is scope for a removal goal, and the longer it waits the more there is to
remove. Note also that this is sentence generation, not grammar — `트럭는탈것이야` has the wrong particle
(`트럭은`), no particle agreement is implemented, and the evaluator builds its targets by the same rule.

**Global top-k activity selection removed** (`test_homeostatic_firing.py`). Ranking a region into a fixed
active quota is something no neuron can do, and it was the most pervasive of the scaffolds the two-track
split disqualifies. It is replaced by what this project's own design principle already prescribes: each
neuron owns a firing threshold and retunes it from its own firing history alone — no region rank, no
region maximum, no region sum — so the active count is whatever crosses threshold and varies per input.

| probe | baseline (global top-k) | homeostatic threshold |
|---|---:|---:|
| first QA (boundary-free syllable) | 8/8 | 8/8 |
| retention after later learning | 7.25/8 | **8/8** |
| new learning | 3.75/4 | **4/4** |
| perfect seeds (continual) | 60% | **6/6** |
| branches recruited | 227 | **96** |
| answer sentence, exact | 76/80 | **80/80** |
| answer sentence, perfect seeds | 7/10 | **10/10** |

The fixed top-k control is also rescored inside the same run on the same seeds (`fixed_topk`, 76/80), so
the +4/80 contribution is not a comparison against numbers from an earlier commit. The other ablations
are unchanged: no replay 0/80, no trace 0/80, no query gate 40/80.

Every probe held its floor and three improved, which was not the expectation for a removal goal. Two of
my diagnoses along the way were wrong and measurement corrected them. Homeostasis first looked like it
destroyed the category geometry; in fact the control loop was misbuilt — the threshold moved ten times
faster than the firing-rate estimate it reads, so it overshot into silence (density 0.025), while a
tenfold slower actuator never controlled anything and the region saturated (density 0.95). With the loop
matched, the category gap came out *wider* than the baseline's (+0.44/+0.54 against +0.26/+0.14), so the
premise that top-k was carrying the geometry was itself wrong. Second, I had built homeostasis as a
rate equalizer, which flattens the selectivity the geometry is made of; cortical rates are broadly
distributed and this kind of homeostasis is a slow stabilizer against silence and runaway, not an
equalizer, so it now acts only outside a band.

The sharper finding came from the regression it caused. Homeostatic firing alone broke continual
learning (retention 90.6% → 81.25%, perfect seeds 60% → 0%) while *improving* everything else. A firing
threshold is a fast variable shared by every memory its neuron takes part in, and it had no warm/cold
separation — later learning retuned it freely and moved the R activity older concepts had been
consolidated against. **That was a hole in the stability-plasticity solution of the previous goal, found
only by removing this scaffold.** Giving intrinsic excitability the same warm→cold consolidation the
synapses use closed it, and retention went to 100%.

**Audit:** `global_sparse_activity_competition` leaves the remaining list and homeostatic firing plus
intrinsic-excitability consolidation enter the satisfied list. Remaining scaffolds go from nine to
**five**. The count had risen 4 → 7 → 9 across the previous three goals; this is the first net decrease,
and no new global machinery was added to get it. Of the four devices the two-track split disqualifies,
two are now resolved; exact teacher error and global seed balancing remain.

Cost: R density rises from 0.18 to 0.405. Homeostasis pins a neuron's time-averaged rate, not the
fraction of neurons active per input — the two are different quantities that agree only on average — so
locality was bought at the price of a sparsity guarantee. Sparsity is not an objective here (see the top
of this file), and the assemblies did not become less distinct; the category gap widened. The band and
control rates are fitted at this scale and were not tested for transfer.

```bash
python3 -m unittest -v test_spatial_connectome.py test_homeostatic_firing.py
python3 spatial_connectome.py
python3 context_branch_probe.py --seeds 20 --rounds 180 --verify
python3 category_generalization_probe.py --seeds 20 --verify --quiet
python3 category_memory_output_probe.py --seeds 20 --verify --quiet
python3 dialogue_qa_probe.py --seeds 10 --verify --quiet
python3 syllable_chunk_probe.py --seeds 10 --verify --quiet
python3 syllable_sentence_probe.py --seeds 10
python3 -m unittest -v test_syllable_chunk_dialogue.py test_syllable_sentence_dialogue.py
```

### The current performance scaffold

```
input (identity)   reasoning (composition/routing)   output (generation)
sparse code    →   fixed-random routing          →   Hebbian/local readout
(k-hot, or         (reservoir attention,             (log readout,
 learned            mixed selectivity)                unigram backoff)
 embedding)        ★no learning needed, fixed random★
```

- **Input code**: k-hot sparse. Either random (sensory stability) or a **structured code** based on context
  profiles (tokens with similar contexts share dimensions → generalization). The structured code is
  significantly better at large vocab.
- **Reasoning/routing**: mixed-selectivity expansion or dot-product routing. **Key finding — leaving the routing
  projection fixed random (reservoir; Maass, Jaeger) beats a learned routing.** Composition (t-1,t binding) uses
  order-preserving projections (P1≠P2) with a high threshold (AND), or softmax routing.
- **Output**: `log(relu(·)+ε) + α·unigram`. Weber-Fechner log compression plus baseline-excitability backoff.
- **Learning**: local delta (readout) + feedback alignment/DFA (embedding). Adam/backprop are a stepping stone
  for measuring the ceiling.

The `Embedding`, `P1/P2`, `W_bi/W_co`, and separate Q/K/V heads in this scaffold are devices I decomposed in
order to diagnose per-mechanism performance. They are not the same as the original single-connectome
implementation, and folding these results into the target structure without losing them is the next main line.

---

## 3. Version arc (summary)

Each stage is a separate file. Detailed diagnoses are in the wiki.

| version | core | wikitext2 vocab2000 val ppl |
|---|---|---|
| v1/v2 | gradient + symmetric Hebbian | collapse (all patterns merge) |
| v3–v8 | STDP, working memory, homeostasis, R-STDP | repeated collapse/explosion |
| v9 | delta rule (LTP+LTD, negative learning) | 936 (first stable, but no semantic emergence) |
| v11 | pure Hebbian counting + log readout | 348 |
| — | count bigram + unigram backoff | **63** |
| v12 | 3-way split dynamics (recurrent reasoning) | single-attractor collapse (identity lost) |
| v13 | reasoning = composition (mixed selectivity), static | vocab10k: random 486 → structured 311 → +composition 302.7 |
| v14 | learned embedding + nonlinear composition | backprop **55.4** / bio(sign) 64.5 / FA 69 |
| v15 | isolating the attention lever | additive 61.8 → attention 58.6 → frozen-routing 56.8 |
| — | (reference) same-size transformer | 0.42M **53**, 0.9M **44** |
| — | search for a softmax replacement (`multiplicative_gate.py`): multiplicative gate, comparison gate, matrix mixing, divisive norm, lateral inhibition (with dt) | the whole normalization family (gate/cmpgate/matmul/divnorm/lateral) sits at **59.4–62.7 (ceiling confirmed)** — none reach attn 57.4 |
| — | phase (binding-by-synchrony: phase-coherence gating, not normalization) | beta=2.0 **58.4, best** — passes the normalization-family ceiling (59.4), narrows the gap to attn (57.4) from 2p to 1p |
| — | phase2 (Kuramoto mutual coherence, population coupling across the whole window) | it=1–8 all 58.8–59.0 — **no improvement**, slightly worse than phase alone (58.4) |
| — | order_probe.py: structural check of window order sensitivity | **attn/divnorm/lateral/phase/phase2 are all order-blind** (swapping memory slots leaves the output identical, Δ=0) — only add/gate/cmpgate/matmul see order. The attn family's edge turns out to be "competitive content-based selection", not "using order" |
| — | phase_pos (theta-gamma order code: fixed positional phase + content phase combined) | structurally order-sensitive (Δ=0.58) confirmed — but beta=1.0/2.0/4.0 give **59.7/60.3/61.1, all worse than phase alone (58.4)**. Adding order information hurts on this benchmark (K=8 vocab2000, bigram-dominated) |

---

## 4. Key findings (the diagnostic chain)

1. **Two kinds of collapse, separated.** ① Sign collapse (all-negative): the predictive-coding target is mostly
   0, so the error is entirely negative → vocab-independent. Solved by **err-centering** (zero-mean error; built
   into the cross-entropy gradient). ② Attractor collapse (single fixed point): a classical Hopfield-style
   implementation storing the whole vocab as fixed points in a recurrent net exceeded capacity. **This refutes
   the token=fixed-point assumption, not the single connectome as a whole.**
2. **Representation learning is possible bio-locally.** Feedback alignment (no backprop) learns embeddings and
   does not collapse (vocab2000: FA 83 → decreasing, backprop 62.9). The cold-warm-hot scaffold (structural
   prior + FA learning) beats random init. → "credit assignment = hard wall" is refuted. What remains is the
   performance gap.
3. **Structured feedback > random.** Hidden feedback: random-FA 70 < sign-concordant 64.5 < backprop 60.4.
   Sign-consistent (E/I-consistent) feedback closes most of the gap. Pure random is a lower-bound proof; a real
   brain would use structured feedback (in the predictive-coding direction).
4. **Binding is the remaining lever.** Additive combination cannot capture order or interaction, so it ceilings
   at 58. Softmax routing (attention) beats additive (58.6 vs 61.8). But pairwise Hadamard (product as feature)
   fails — the product has to be used as a *score*. Fully closing the gap (→53) needs a transformer stack = not
   bio.
5. **Only the form of simple dialogue is achieved.** With a large vocab (removing the `<unk>` flood) and dialogue
   data (empathetic dialogues, turn markers), the frozen-routing model responds in an empathetic register. But it
   fails fact, inference, and concrete-instruction probes, so I judged it a generic empathizer. The fully-bio
   (DFA) version keeps the form but pays in quality.
6. **Window competition without softmax still does not close the gap.** Non-commutative combination
   (multiplicative gate, comparison gate, matrix mixing; 60.2–60.5) does preserve order but leaves the gap to
   attn unchanged — the cause is not non-commutativity but the absence of competitive normalization across the
   whole window. Divisive normalization (Carandini & Heeger, measured exponent n≈2) is the best non-softmax
   (59.4) but still ~2p short of attn (57.4). **This is measured evidence that even the brain's exact
   normalization formula falls short of softmax's computational power** — tuning the value (n=1/4) does not beat
   the measured one (n=2). Iterative lateral inhibition (multi-step convergence instead of a static formula) is
   the remaining candidate.

---

## 5. Evaluation (standard metrics + efficiency)

**Dialogue held-out val perplexity** (empathetic_dialogues, ~1.15M params, matched size):

| model | training | val ppl |
|---|---|---|
| Ours (frozen-attn, backprop E/head) | 5 epochs (converged) | **96.8** |
| same-size transformer | 10 / 20 / 25 epochs | 128.6 / 92.9 / 87.2 |
| Ours (fully-bio, DFA) | 15 epochs | 160.5 (still falling) |

- **Efficiency edge:** ours reaches 96.8 in 5 epochs; the transformer needs 20 (≈4–5× learning efficiency). The
  ppl *ceiling* is lower for the transformer (87.2), but *performance per unit of compute* favors ours — which
  connects to the "route around GPT's waste" thesis.
- **The cost of going fully bio:** DFA (no backprop) is at 160 (15ep, ~1.7× gap, still falling); it keeps the
  empathetic register but is rougher than backprop (more `<unk>`, more repetition). Both attempts to narrow the
  gap failed: (a) structured feedback (sign-concordant, direct path) 169.8 ≈ random 169.1; (b) routing credit
  through the attention-mix path as well (full-path DFA) 199.1 — *worse*, since every added random feedback
  matrix adds noise rather than signal. → **The gap is not a "missing credit path" but an intrinsic
  approximation limit of random-feedback credit.** Tuning feedback cannot close it; this needs fundamentally
  better bio credit (predictive coding, target propagation, etc. — open problems in the field).
- **Limits of the metric:** ppl is a lookup-biased metric (a count table beats dynamics). It is a sanity check;
  the value of thought/dynamics needs separate axes (long-range, composition, continual learning, internal-
  structure RSA).

**Honest quantification of efficiency** (`efficiency.py`, `dialogue_sparse.py`):
- per-token FLOPs: ours 822k ≈ tf 1.17M (the vocab readout head dominates both; dense, there is no big
  difference).
- The real edge: ① fewer learned parameters (routing is fixed) ② 4–5× faster convergence ③ no backprop (DFA,
  mostly forward).
- **Restoring sparsity (important):** the dialogue model used learned dense embeddings, density 0.69 — a
  departure from the sparse principle. Forcing top-k on the features restores sparsity: at density 0.06
  (k=8/128, around the brain's ~5% firing rate) dialogue survives, at a cost of ppl 105→227 (~2.2×). Features at
  6% activity → ~11× less compute in the dominant head = the sparse-coding energy/accuracy tradeoff (the
  neuromorphic value). The efficiency edge comes not from being dense but from **forced sparsity + learning
  efficiency**.
- **Sparsity as a genetic prior + threshold firing (`dialogue_adaptive_sparse.py`):** instead of hard top-k,
  **threshold firing** (input>θ, with θ held by homeostasis at a target firing rate ρ) = neuron-style firing with
  a naturally varying k. Density 0.10, ppl 207 — **better than fixed top-k (228)** (strong units survive, only
  weak ones die). k varies per input (std 0.06) but does not correlate with difficulty (+0.015). This implements
  the design principle "sparsity does not emerge from learning; it is given as a genetic prior (a homeostatic
  threshold)".

**Dialogue samples** (backprop model, `gen_saved.py`):

```
USER: my dog is sick
BOT : i would be so upset about it.
USER: i am scared about my exam
BOT : oh wow! i hate that is so i hope he is very painful.
```

---

## 6. Methodological principles

- **Diagnose first, mechanism second.** Before implementing an idea, pin down the root of the failure by
  measurement (the repeated lesson of this project; most of the sophisticated learning rules broke basic
  statistics).
- **Bio = a stepping stone.** Once the structure is made brain-consistent, backprop is fine for checking the
  ceiling. The final goal is bio-local.
- **Verify against the source directly.** Confirm performance numbers and bugs by running the code (e.g. the MPS
  `multinomial` bug returning out-of-range indices, indexing off-by-one, a ~1.5% optimistic bias from the val
  subset — all caught by running things).
- **Two kinds of formula transplant.** (a) Keep the form and tune only the constants (e.g. the divnorm n-sweep —
  the measured brain value n=2 beat the tuned ones) vs (b) drop the form and reproduce only the meaning, simply
  (e.g. v11 log-Hebbian). Neither is always superior — each case needs measurement. The reconstruction is
  top-down, but it does not transplant the brain's exact formulas wholesale (this is what separates it from GCI's
  bottom-up / physical-primitive principle).
- **Check the scale a formula applies at.** A formula aimed at the brain in general (a cortical local circuit)
  can collapse even with the right form if it is applied at the wrong pool scale (v9's divisive readout divided
  by the mean over all N units instead of a local pool). Before transplanting, check what pool/scale the formula
  originally demands.
- **Separate formula-level from structure-level bio grounding.** The items above are about the equations of a
  single operation (formula level). Separately, the input/reasoning/output 3-way split (Fedorenko, language vs
  reasoning networks) and the cold-warm-hot multi-timescale scheme (CLS, Benna-Fusi) are structure-level bio
  grounding and hold independently — dropping one line of inquiry (e.g. replacing softmax) does not affect the
  other.

---

## 7. Limits and open problems

- **Bio-local attention.** The routing advantage was captured with fixed random projections (reservoir), but
  softmax-dot is not fully bio. Non-commutative combination, divisive norm (including the measured exponent, best
  at 59.4), and lateral inhibition were all tried, and a ~2p gap to attn (57.4) persists — the normalization
  family looks close to its ceiling. Axes other than normalization, like the brain's binding (theta-gamma phase
  code, dendritic coincidence detection), are still untried.
  **Four branches (organized 2026-07-08)**:
  1. Revive dynamics — iterative convergence (recurrence) instead of a static formula, applied after separating
     the local pool (single-neuron/cluster scale) from the brain-wide formula scale (connects to design-principle
     ⑧, the scale mismatch principle).
  2. Keep only the meaning of the formula and simplify it (the ⑦-b family, e.g. the v11 log-Hebbian approach).
  3. Just accept softmax as a non-bio exception — once the normalization family's ceiling (~59) is confirmed,
     return to this project's main line (dialogue/efficiency axes).
  4. Phase code (theta-gamma) — everything so far (divnorm/lateral/gate) has been the "competitive
     normalization" family. Splitting slots by oscillation phase to encode order is a completely different,
     non-normalization axis. Outcome unpredictable, needs new code from scratch.

  Note: 1 and 2 are not an exclusive fork — divnorm (n=2, 59.4) was already both at once (scale-matched to a K=8
  local pool + simplified from the original formula with relu/powers), and the lateral+dt experiment was version 1
  extended to iterative convergence.

  **Verdict reached (dt sweep, n=2/beta=0.3 fixed, Euler leaky integration)**: dt=0.3→59.8 (best) | dt=0.5→62.1 |
  dt=0.7→61.6 | dt=1.0→60.9. **None beat divnorm 59.4.** Branch 1+2 (dynamics) is blocked here — the
  normalization family's ceiling (gate/cmpgate/matmul/divnorm/lateral, all of it) is **finally fixed at ~59.4**.

  **Branch 4 (phase code) result — a breakthrough.** Implemented the `phase` mode: convert the content-match
  score to a phase φ∈[0,π/2] (β=steepness, high match→φ→0, in-phase), and gate by the coherence gain w=cos(φ)²
  (Malus's law) — **with no normalization over the whole window** (this removes the "competitive normalization"
  property that softmax/divnorm/lateral all shared; the analogy is Fries 2005 communication-through-coherence).
  β sweep: 0.5→58.5 | 1.0→58.9 | **2.0→58.4 (best)** | 4.0→62.0 (too steep, becomes a binary gate and collapses).
  **The first time the normalization family's ceiling (59.4) is passed — the gap to attn (57.4) narrows from 2p
  to 1p, the best non-attn record so far.** This suggests softmax's core may not be "normalization" but
  "match-based coherence (unnormalized)" itself.

  **phase2 (Kuramoto mutual coherence) result — no improvement.** Implemented and ran real population coupling
  where window tokens pull each other's phases (`coup_ij=tanh(k_i·k_j/√d)` + Kuramoto update, sweeping iters).
  it∈{1,3,5,8} all **stall at 58.8–59.0** — slightly worse than phase alone (58.4). Adding coupling does not help.

  **Structural check (`order_probe.py`, new) — the bigger finding.** With no training, purely structurally: swap
  the window's memory slots (positions 1..K-1) and see whether the output changes. **attn (including real softmax
  attention), divnorm, lateral, phase, and phase2 are all order-blind** (Δlogit=0.000000, identical down to
  floating point) — the attn family in this code has no positional encoding to begin with, so mix=Σw_j·v_j is a
  symmetric sum over j (a measured reproduction of exactly why a transformer must use a separate positional
  embedding). Only add/gate/cmpgate/matmul (per-slot weights, recurrence) see order. **Reinterpretation: the attn
  family's edge — over divnorm, and over gate — was not "it uses order better" but "a difference in competitive
  content-based selection".** phase2's premise that "coherence = real binding" also needs rechecking from this
  view (order information never enters at all). **New lever candidate**: add a positional signal to k_j (fixed
  sinusoidal etc., consistent with the genetic-prior principle) — no attn-family variant here has used that axis,
  so both attn (57.4) and phase (58.4) may have room to go lower. Untried.

  **Lever attempt — phase_pos, failed.** A theta-gamma serial-order code (Lisman & Idiart 1995): add a fixed
  phase offset per slot position j, `pos_j=(π/2)·j/(K-1)` (j=0="now" ~ j=K-1=oldest), to the content-match phase,
  binding "what" and "when" into one gate. order_probe confirms structural order-sensitivity (Δ=0.58; phase and
  phase2 are 0). **But the training result: β=1.0→59.7 | β=2.0→60.3 | β=4.0→61.1, all worse than phase alone
  (58.4).** A fixed linear positional decay seems to act as noise rather than useful signal on this benchmark
  (K=8, vocab2000, bigram-dominated) — it could be that the order information went in but the decay shape/scale is
  wrong, or that this task leans on local statistics more than order in the first place (repeatedly confirmed
  across the arc); both remain, undistinguished. **Final conclusion for this session: phase (β=2.0, 58.4) stands
  as this arc's best non-attn record**; phase2 and phase_pos both failed to improve on it.

  **A separate axis (structure level, design-principle ⑨) — independent of the four branches above, dropping one
  does not block the other**: the input/reasoning/output 3-way split (already implemented, grounded in
  Fedorenko's language-vs-reasoning network separation) and the cold-warm-hot multi-timescale scheme (partially
  implemented; the warm→cold consolidation stage is not) are separate considerations with their own brain
  grounding — replacing softmax is a formula-level experiment and does not touch this side.
- **Returning to dynamics.** Recurrent reasoning (v12) is on hold over attractor capacity. Multi-timescale
  (Benna-Fusi) and consolidation are candidate prescriptions.
- **Neuromodulation (dopamine) not applied.** Three-factor (pre×post×dopamine = reward/error gating) attaches to
  feedforward learning too (no dynamics needed). It is the next candidate for dialogue quality (bio-RLHF).
- **The DFA performance gap.** Fully-bio learning is at 2× the ppl of backprop. Narrow it with structured
  feedback and predictive coding.

---

## 8. References

**Local learning / credit assignment**
- Hebb, D.O. (1949). *The Organization of Behavior.* — Hebbian learning, cell assembly.
- Bi, G. & Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons. *J. Neurosci.* — STDP.
- Lillicrap, T. et al. (2016). Random synaptic feedback weights support error backpropagation. *Nat. Commun.* — **Feedback Alignment**.
- Nøkland, A. (2016). Direct Feedback Alignment provides learning in deep neural networks. *NeurIPS* — **DFA**.
- Liao, Q., Leibo, J., Poggio, T. (2016). How important is weight symmetry in backpropagation? *AAAI* — sign-concordant feedback.
- Kolen, J. & Pollack, J. (1994). Backpropagation without weight transport. *IEEE ICNN*; Akrout et al. (2019) — weight mirror.
- Guerguiev, J., Lillicrap, T., Richards, B. (2017). Towards deep learning with segregated dendrites. *eLife* — dendritic error.
- Payeur, A. et al. (2021). Burst-dependent synaptic plasticity. *Nat. Neurosci.*

**Sparsity / mixed selectivity / reservoirs**
- Maass, W., Natschläger, T., Markram, H. (2002). Real-time computing without stable states (Liquid State Machines). *Neural Comput.*
- Jaeger, H. (2001). Echo State Networks. — reservoir computing.
- Rigotti, M. et al. (2013). The importance of mixed selectivity in complex cognitive tasks. *Nature.*
- Marr, D. (1969); Albus, J. (1971) — cerebellar granule cell expansion code / supervised learning.
- Quian Quiroga, R. et al. (2005). Invariant visual representation by single neurons (concept cells). *Nature.*

**Prediction / neuromodulation / homeostasis**
- Rao, R. & Ballard, D. (1999). Predictive coding in the visual cortex. *Nat. Neurosci.*
- Friston, K. (2005). A theory of cortical responses. *Phil. Trans. R. Soc.*
- Schultz, W., Dayan, P., Montague, P. (1997). A neural substrate of prediction and reward (dopamine RPE). *Science.*
- Benna, M. & Fusi, S. (2016). Computational principles of synaptic memory consolidation. *Nat. Neurosci.* — multiple timescales.
- Turrigiano, G. (2008). The self-tuning neuron: synaptic scaling / homeostasis. *Cell.*

**Language / brain metrics**
- Fedorenko, E. et al. — separation of the language network vs the multiple-demand (reasoning) network.
- Schrimpf, M. et al. (2021). The neural architecture of language: integrative modeling. *PNAS* — surprisal ↔ language-network fMRI/N400.
- Lisman, J. & Jensen, O. (2013). The theta-gamma neural code. *Neuron.*

**Comparison targets**
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS* — transformer (the title is an homage).
- Merity, S. et al. (2016). WikiText-2. — benchmark data.
- Rashkin, H. et al. (2019). EmpatheticDialogues. — dialogue data.

---

## 9. Files and reproduction

```
model_v9.py / v12.py / v13.py    architecture snapshots (collapse/composition diagnoses on record)
train_v13.py                     v13 training/eval (structured code + composition + Hebbian)
structured_code.py               structured code experiment (#11)
data.py                          wikitext2 loader (pad=0, unk=1)
```

> The experiment scripts from this session (v14/v14b/v14ctx/v15/v15b/v15c, replearn/rl2, dialogue
> dlg/gen_saved/dlg_fa, benchmarks dlg_bench/fairtf) are still to be cleaned up for reproduction. Dialogue model
> checkpoint: `dlg_model.pt`.

```bash
python3 train_v13.py     # compositional reasoning (vocab10k)
# dialogue: train on empathetic_dialogues → generate responses (dlg scripts)
```

Key coordinates (vocab2000): `uniform 2000 | unigram 125 | bigram 78 | count+backoff 63 |
v14-backprop 55 | v14-bio(sign) 64 | attention 58 | tf 53 | tf-0.9M 44`.
Dialogue (vocab6000, 1.15M): `ours 96.8(5ep) | tf 87(25ep) | ours-DFA 192(8ep, fully bio)`.
