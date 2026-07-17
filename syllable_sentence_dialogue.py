"""Answer-sentence generation for boundary-free syllable questions.

``syllable_chunk_dialogue`` answers with one O-token.  Here the answer is an
ordered syllable emission produced inside the connectome, with no external
autoregressive decoder, no terminal symbol and no candidate list.

An answer sentence draws on two pathways because its parts have different
origins.  The content is closed-class and generalizes from category prototypes
through R.  The entity's name is open-class -- a held-out entity never receives
a sentence target -- so it can only come from that entity's own chunk driving
the syllable assemblies it was recruited from.  The two are kept apart: the
replayed name never enters the emission trace.

Serial order comes from O-local dendritic branches whose presynaptic sources
include a decaying trace of the O assemblies just emitted.  A branch normalizes
its meaning stream and its order stream separately and multiplies them, so the
two do not compete for one budget, and the order stream alone decides when the
sentence is over.

Two tracks share the interface:

* ``FunctionalSentenceDialogue`` may align the target sentence against the
  question and unpack a chunk into its syllables.  Both are global devices and
  are declared as the capability ceiling, not as biological mechanisms.
* ``ConnectomeSentenceDialogue`` extends the single connectome of
  ``ConnectomeSyllableDialogue`` and may use neither.  ``locality_audit``
  exposes what it still leans on.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import torch

from spatial_connectome import OUTPUT
from syllable_chunk_dialogue import (
    ConnectomeSyllableDialogue,
    FunctionalSyllableDialogue,
    syllable_sequence,
)


@dataclass(frozen=True)
class SentenceLocalityAudit:
    """What the sentence track satisfies and what it still leans on.

    The ``global_`` fields are scaffolds, kept honest rather than hidden: the
    constituent delta sweeps every learned chunk, membership is read by matching
    against whole syllable assemblies, and the sentence target is clamped. They
    are not counted as biological locality.
    """

    local_chunk_constituent_delta: bool
    name_replay_from_own_chunk: bool
    separate_repetition_and_semantic_pathways: bool
    two_stream_dendritic_coincidence: bool
    order_gated_termination: bool
    single_connectome_memory_and_output: bool
    no_autograd_or_weight_transport: bool
    global_constituent_sweep: bool
    global_syllable_assembly_decoding: bool
    global_teacher_clamped_sentence: bool
    global_window_enumeration: bool
    global_fact_episode_intersection: bool
    global_sparse_activity_competition: bool
    global_seed_balancing: bool
    global_output_argmax: bool

    def as_dict(self) -> dict[str, bool]:
        return asdict(self)


def emission_trace(
    patterns: Sequence[torch.Tensor], *, decay: float
) -> torch.Tensor:
    """Decaying trace of recently emitted assemblies, newest weighted most."""
    trace = torch.zeros_like(patterns[0]) if patterns else torch.zeros(0)
    for age, pattern in enumerate(reversed(patterns)):
        trace = trace + (decay ** age) * pattern
    return trace


class ConnectomeSentenceDialogue(ConnectomeSyllableDialogue):
    """Ordered syllable emission from the existing single connectome.

    An O neuron's branch sees its own R sources and the O units still active
    from the last few emissions.  Recruitment, novelty and maturity rules are
    the ones already audited for the single-token answer.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        output_units: int = 256,
        trace_decay: float = 0.5,
        trace_length: int = 3,
        max_emission: int = 14,
        constituent_gate: float = 0.55,
        use_replay: bool = True,
        use_trace: bool = True,
    ):
        super().__init__(seed=seed, output_units=output_units)
        self.trace_decay = trace_decay
        self.trace_length = trace_length
        self.max_emission = max_emission
        self.constituent_gate = constituent_gate
        self.use_replay = use_replay
        self.use_trace = use_trace

    def register_syllable_output(self, texts: Iterable[str]) -> None:
        """Every syllable is an emittable motor assembly."""
        vocabulary = sorted(
            {syllable for text in texts for syllable in syllable_sequence(text)}
        )
        self.connectome.register_vocabulary(vocabulary)

    def _constituent_drive(self, entity: tuple[str, ...]) -> dict[str, float]:
        """Which syllable assemblies the active entity chunk drives."""
        code = self.concept_patterns[entity]
        normalized = code / code.square().sum().sqrt().clamp_min(1e-8)
        drive = normalized @ self.chunker.constituent
        scores = {}
        for syllable, syllable_code in self.chunker.bank.codes.items():
            weight = float(syllable_code.sum())
            if weight > 0:
                scores[str(syllable)] = float(drive @ syllable_code) / weight
        return scores

    def _replay_syllable(
        self, entity: tuple[str, ...], spoken: Sequence[str]
    ) -> str | None:
        """Next syllable of the entity's own name, ordered by local transitions.

        The category path cannot supply an open-class name: held-out entities
        never receive a sentence target.  The chunk instead drives its own
        constituent syllables, and the transition synapses that built the chunk
        order them.  Already-spoken syllables are refractory.
        """
        # The delta rule drives a constituent syllable assembly toward its own
        # full activation and everything else toward zero, so membership is an
        # absolute question -- is this assembly driven at more than half? -- and
        # not a fraction of whichever syllable happens to lead this chunk.
        drive = self._constituent_drive(entity)
        members = [
            syllable
            for syllable, value in drive.items()
            if value > self.constituent_gate
        ]
        remaining = [item for item in members if item not in spoken]
        if not remaining:
            return None
        # The transition synapses that built the chunk say what follows what.
        # They are plain Hebbian, so overlapping syllable assemblies leave a
        # spurious floor and a fixed threshold misreads it; rank instead. The
        # syllable least driven by any other unspoken member comes first.
        def incoming(item: str) -> float:
            return max(
                (
                    self.chunker._pair_strength(other, item)
                    for other in remaining
                    if other != item
                ),
                default=0.0,
            )

        return min(remaining, key=incoming)

    def _emission_context(
        self, text: str, emitted: Sequence[torch.Tensor], *, controlled: bool = True
    ) -> torch.Tensor:
        bound = self.bound_state(text, controlled=controlled)
        recent = list(emitted)[-self.trace_length :] if self.use_trace else []
        if recent:
            trace = emission_trace(recent, decay=self.trace_decay)
            output = self.connectome.region == OUTPUT
            bound[output] = trace[output]
        return bound

    def _branch_syllable(
        self, context: torch.Tensor
    ) -> tuple[str | None, float]:
        """Emit while the order stream still recognizes the present trace.

        The full coincidence cannot decide this.  Its meaning stream is weaker
        for a held-out entity than for the prototype that recruited the branch,
        so a threshold on the product confuses "this name is new" with "this
        sentence is over".  The order stream carries no entity information: it
        reads 1.0 at every syllable the sentence still owes and collapses once
        nothing expects the trace, which ends emission with no terminal symbol
        and no length limit.  Recruitment already calls that threshold the point
        where a branch covers a context.
        """
        emission = self.connectome.dendritic_sequence_emission(context)
        if not emission:
            return None, 0.0
        token, (_, order) = max(emission.items(), key=lambda item: item[1][0])
        recognized = self.connectome.config.dendritic_branch_novelty_threshold
        return (token, order) if order >= recognized else (None, order)

    def _next_syllable(
        self,
        text: str,
        entity: tuple[str, ...],
        emitted: Sequence[torch.Tensor],
        spoken: Sequence[str],
        *,
        controlled: bool = True,
    ) -> tuple[str | None, str]:
        """The name replay speaks first; the category path continues.

        The two pathways are separate: the replayed name never enters the
        emission trace, so a frame branch keyed on the trace stays about what
        was said, not about which name said it, and generalizes to a held-out
        entity whose name the category path has never produced.
        """
        replay = self._replay_syllable(entity, spoken) if self.use_replay else None
        if replay is not None:
            return replay, "replay"
        context = self._emission_context(text, emitted, controlled=controlled)
        token, _ = self._branch_syllable(context)
        return token, "branch"

    def teach_sentence(
        self,
        text: str,
        sentence: str,
        *,
        learn_control: bool = True,
    ) -> None:
        """Teacher-clamped emission; recruit only where the circuit is wrong."""
        if learn_control:
            self.learn_query_control(text)
        entity = self._entity_pattern(text)
        emitted: list[torch.Tensor] = []
        spoken: list[str] = []
        for syllable in syllable_sequence(sentence):
            produced, source = self._next_syllable(text, entity, emitted, spoken)
            spoken.append(syllable)
            if source == "replay" and produced == syllable:
                continue
            if produced != syllable:
                context = self._emission_context(text, emitted)
                self.connectome.learn_dendritic_sequence(
                    context, self.connectome.output_pattern(syllable)
                )
            emitted.append(self.connectome.output_pattern(syllable))

    @staticmethod
    def locality_audit() -> SentenceLocalityAudit:
        return SentenceLocalityAudit(
            local_chunk_constituent_delta=True,
            name_replay_from_own_chunk=True,
            separate_repetition_and_semantic_pathways=True,
            two_stream_dendritic_coincidence=True,
            order_gated_termination=True,
            single_connectome_memory_and_output=True,
            no_autograd_or_weight_transport=True,
            global_constituent_sweep=True,
            global_syllable_assembly_decoding=True,
            global_teacher_clamped_sentence=True,
            global_window_enumeration=True,
            global_fact_episode_intersection=True,
            global_sparse_activity_competition=True,
            global_seed_balancing=True,
            global_output_argmax=True,
        )

    def generate(self, text: str, *, controlled: bool = True) -> str:
        entity = self._entity_pattern(text)
        emitted: list[torch.Tensor] = []
        spoken: list[str] = []
        for _ in range(self.max_emission):
            token, source = self._next_syllable(
                text, entity, emitted, spoken, controlled=controlled
            )
            if token is None:
                break
            spoken.append(token)
            if source != "replay":
                emitted.append(self.connectome.output_pattern(token))
        return "".join(spoken)


class FunctionalSentenceDialogue:
    """Global ceiling: aligned sentence frames plus chunk unpacking.

    Declared global devices: aligning the target sentence against the question
    to locate the entity span, unpacking a learned chunk into its ordered
    syllables, and a frame memory keyed by the answered content token.
    """

    def __init__(self, *, seed: int = 0):
        self.base = FunctionalSyllableDialogue(seed=seed)
        self.frames: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)

    def register_syllable_vocabulary(self, texts: Iterable[str]) -> None:
        self.base.register_syllable_vocabulary(texts)

    def register_output_vocabulary(self, tokens: Iterable[str]) -> None:
        self.base.register_output_vocabulary(tokens)

    def observe_streams(self, texts: Iterable[str], *, temporal: bool = True) -> None:
        self.base.observe_streams(texts, temporal=temporal)

    def finalize_chunks(self) -> None:
        self.base.finalize_chunks()

    def observe_fact(self, text: str, properties: Sequence[str], **kwargs) -> None:
        self.base.observe_fact(text, properties, **kwargs)

    def teach_answer(self, text: str, target: str, **kwargs) -> None:
        self.base.teach_answer(text, target, **kwargs)

    def _entity_chunk(self, text: str) -> tuple[str, ...] | None:
        """The most semantically loaded chunk in the question.

        Fact learning gives entity chunks strong outgoing semantic synapses
        while shared question wording stays weak, so the entity is identified
        by connectivity rather than by length or by a supplied span.
        """
        best: tuple[str, ...] | None = None
        best_strength = 0.0
        for pattern, code, _ in self.base.chunker.matched_codes(text):
            strength = float(
                (code @ self.base.semantic_weights).square().sum().sqrt()
            )
            if strength > best_strength:
                best, best_strength = pattern, strength
        return best

    def teach_sentence(self, text: str, sentence: str, target: str) -> None:
        """Abstract the sentence into an entity slot plus a content frame."""
        entity = self._entity_chunk(text)
        answer = syllable_sequence(sentence)
        content = syllable_sequence(target)
        frame: list[str] = []
        index = 0
        while index < len(answer):
            if entity and tuple(answer[index : index + len(entity)]) == entity:
                frame.append("<entity>")
                index += len(entity)
            elif tuple(answer[index : index + len(content)]) == content:
                frame.append("<content>")
                index += len(content)
            else:
                frame.append(answer[index])
                index += 1
        self.frames[target][tuple(frame)] += 1

    def generate(self, text: str, *, temporal: bool = True, controlled: bool = True) -> str:
        target, _, _ = self.base.answer(text, temporal=temporal, controlled=controlled)
        if not self.frames[target]:
            return ""
        frame = self.frames[target].most_common(1)[0][0]
        entity = self._entity_chunk(text) or ()
        pieces: list[str] = []
        for slot in frame:
            if slot == "<entity>":
                pieces.extend(entity)
            elif slot == "<content>":
                pieces.extend(syllable_sequence(target))
            else:
                pieces.append(slot)
        return "".join(pieces)
