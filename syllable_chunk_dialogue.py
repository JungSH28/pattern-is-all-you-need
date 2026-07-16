"""Boundary-free syllable assembly dialogue models.

The public API accepts only continuous Korean text and supervised fact/answer
targets.  It never accepts word boundaries, entity identifiers, answer
candidates, or query-intent labels.

Two tracks share the same sparse-assembly interface:

* ``FunctionalSyllableDialogue`` uses global substring statistics and global
  outer-product memories as a capability ceiling.
* ``BioLocalSyllableDialogue`` forms temporal assemblies from locally updated
  pre/post transition synapses and uses local coactivity/error updates for
  semantic and output synapses.  Remaining implementation scaffolds are
  exposed by ``locality_audit``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
import unicodedata
from typing import Iterable, Sequence

import torch

from spatial_connectome import SUBSTRATE, ConnectomeConfig, SpatialConnectome


def syllable_sequence(text: str) -> tuple[str, ...]:
    """Mechanical Unicode normalization only; no linguistic segmentation."""
    normalized = unicodedata.normalize("NFKC", text)
    return tuple(character for character in normalized if not character.isspace())


class SparseAssemblyBank:
    """Stable balanced sparse assemblies without an embedding table."""

    def __init__(self, n_units: int, assembly_size: int, seed: int):
        self.n_units = n_units
        self.assembly_size = assembly_size
        self.generator = torch.Generator().manual_seed(seed)
        self.usage = torch.zeros(n_units)
        self.codes: dict[object, torch.Tensor] = {}

    def code(self, item: object) -> torch.Tensor:
        if item not in self.codes:
            tie_break = torch.rand(self.n_units, generator=self.generator)
            units = torch.argsort(self.usage + 1e-3 * tie_break)[: self.assembly_size]
            code = torch.zeros(self.n_units)
            code[units] = 1.0
            self.codes[item] = code
            self.usage[units] += 1.0
        return self.codes[item]


class FunctionalTemporalChunker:
    """Global repeated-substring ceiling for boundary-free syllable streams."""

    def __init__(
        self,
        *,
        n_units: int = 512,
        assembly_size: int = 16,
        max_chunk: int = 6,
        seed: int = 0,
    ):
        self.n_units = n_units
        self.max_chunk = max_chunk
        self.bank = SparseAssemblyBank(n_units, assembly_size, seed)
        self.syllable_bank = SparseAssemblyBank(n_units, assembly_size, seed + 1)
        self.counts: Counter[tuple[str, ...]] = Counter()
        self.left_context: dict[tuple[str, ...], set[str]] = defaultdict(set)
        self.right_context: dict[tuple[str, ...], set[str]] = defaultdict(set)
        self.learned_patterns: dict[tuple[str, ...], float] = {}
        self._texts: list[tuple[str, ...]] = []

    def register_syllables(self, syllables: Iterable[str]) -> None:
        for syllable in syllables:
            self.syllable_bank.code(syllable)

    def observe(self, text: str) -> None:
        sequence = syllable_sequence(text)
        self._texts.append(sequence)
        for length in range(2, min(self.max_chunk, len(sequence)) + 1):
            for start in range(len(sequence) - length + 1):
                pattern = sequence[start : start + length]
                self.counts[pattern] += 1
                self.left_context[pattern].add(sequence[start - 1] if start else "<s>")
                end = start + length
                self.right_context[pattern].add(sequence[end] if end < len(sequence) else "</s>")

    def finalize(self, *, minimum_count: int = 2) -> None:
        self.learned_patterns.clear()
        for pattern, count in self.counts.items():
            if count < minimum_count:
                continue
            branching = math.sqrt(
                len(self.left_context[pattern]) * len(self.right_context[pattern])
            )
            score = math.log1p(count) * len(pattern) ** 1.5 * (1.0 + branching)
            self.learned_patterns[pattern] = score
            self.bank.code(pattern)

    def feature(self, text: str, *, temporal: bool = True) -> torch.Tensor:
        sequence = syllable_sequence(text)
        feature = torch.zeros(self.n_units)
        # Weak syllable trace prevents total silence in temporal ablations.
        for syllable in sequence:
            feature.add_(0.08 * self.syllable_bank.code(syllable))
        if temporal:
            for pattern, score in _winning_patterns(sequence, self.learned_patterns):
                feature.add_(score * self.bank.code(pattern))
        return _sparsify_and_normalize(feature, max_active=96)

    def pattern_code(self, pattern: Sequence[str]) -> torch.Tensor | None:
        key = tuple(pattern)
        return self.bank.codes.get(key)

    def matched_codes(
        self, text: str
    ) -> tuple[tuple[tuple[str, ...], torch.Tensor, float], ...]:
        sequence = syllable_sequence(text)
        return tuple(
            (pattern, self.bank.codes[pattern], score)
            for pattern, score in self.learned_patterns.items()
            if _occurrences(sequence, pattern)
        )

    def winning_codes(
        self, text: str
    ) -> tuple[tuple[tuple[str, ...], torch.Tensor, float], ...]:
        sequence = syllable_sequence(text)
        return tuple(
            (pattern, self.bank.codes[pattern], score)
            for pattern, score in _winning_patterns(
                sequence, self.learned_patterns
            )
        )


class BioLocalTemporalChunker:
    """Temporal chunks formed by local synapses between syllable assemblies.

    Each observed adjacent pair changes only synapses whose presynaptic and
    postsynaptic sensory units were active.  Recurrent path activity then
    supplies the assembly for a repeated multi-syllable pattern.  Enumerating
    finite temporal windows and sparse winner selection remain audited
    simulation scaffolds, not claimed biological mechanisms.
    """

    def __init__(
        self,
        *,
        n_units: int = 512,
        assembly_size: int = 16,
        max_chunk: int = 6,
        learning_rate: float = 0.08,
        seed: int = 0,
    ):
        self.n_units = n_units
        self.assembly_size = assembly_size
        self.max_chunk = max_chunk
        self.learning_rate = learning_rate
        self.bank = SparseAssemblyBank(n_units, assembly_size, seed)
        self.chunk_bank = SparseAssemblyBank(n_units, assembly_size, seed + 1)
        self.transition = torch.zeros(n_units, n_units)
        self.counts: Counter[tuple[str, ...]] = Counter()
        self.learned_patterns: dict[tuple[str, ...], float] = {}
        self.pattern_assemblies: dict[tuple[str, ...], torch.Tensor] = {}

    def register_syllables(self, syllables: Iterable[str]) -> None:
        for syllable in syllables:
            self.bank.code(syllable)

    def observe(self, text: str, *, plastic: bool = True) -> None:
        sequence = syllable_sequence(text)
        codes = [self.bank.code(syllable) for syllable in sequence]
        if plastic:
            for previous, current in zip(codes, codes[1:]):
                pre = previous > 0
                post = current > 0
                # Edge-local Hebbian update followed by source-local saturation.
                self.transition[pre] += self.learning_rate * post.float()
                self.transition[pre] = self.transition[pre].clamp_max(2.0)
        for length in range(2, min(self.max_chunk, len(sequence)) + 1):
            for start in range(len(sequence) - length + 1):
                self.counts[sequence[start : start + length]] += 1

    def _pair_strength(self, previous: str, current: str) -> float:
        pre = self.bank.code(previous) > 0
        post = self.bank.code(current) > 0
        return float(self.transition[pre][:, post].mean().item())

    def finalize(self, *, minimum_count: int = 2) -> None:
        self.learned_patterns.clear()
        self.pattern_assemblies.clear()
        has_repeated_pair = any(
            len(pattern) == 2 and count >= minimum_count
            for pattern, count in self.counts.items()
        )
        if not has_repeated_pair:
            return
        # A fixed local firing threshold: two ordinary coactivations are
        # sufficient.  It does not compare a synapse with a network-wide rank.
        threshold = self.learning_rate * minimum_count * 0.70
        for pattern, count in self.counts.items():
            if count < minimum_count:
                continue
            strengths = [
                self._pair_strength(left, right)
                for left, right in zip(pattern, pattern[1:])
            ]
            cohesion = min(strengths)
            if cohesion + 1e-8 < threshold:
                continue
            score = math.log1p(count) * len(pattern) ** 1.5 * cohesion
            self.learned_patterns[pattern] = score
            # Local transition strength decides whether a chunk forms. Sparse
            # recruitment then gives the stabilized chunk its own assembly;
            # balanced recruitment is explicitly retained in the audit.
            self.pattern_assemblies[pattern] = self.chunk_bank.code(pattern)

    def feature(self, text: str, *, temporal: bool = True) -> torch.Tensor:
        sequence = syllable_sequence(text)
        feature = torch.zeros(self.n_units)
        for syllable in sequence:
            feature.add_(0.08 * self.bank.code(syllable))
        if temporal:
            for pattern, score in _winning_patterns(sequence, self.learned_patterns):
                feature.add_(score * self.pattern_assemblies[pattern])
        return _sparsify_and_normalize(feature, max_active=96)

    def pattern_code(self, pattern: Sequence[str]) -> torch.Tensor | None:
        return self.pattern_assemblies.get(tuple(pattern))

    def matched_codes(
        self, text: str
    ) -> tuple[tuple[tuple[str, ...], torch.Tensor, float], ...]:
        sequence = syllable_sequence(text)
        return tuple(
            (pattern, self.pattern_assemblies[pattern], score)
            for pattern, score in self.learned_patterns.items()
            if _occurrences(sequence, pattern)
        )

    def winning_codes(
        self, text: str
    ) -> tuple[tuple[tuple[str, ...], torch.Tensor, float], ...]:
        sequence = syllable_sequence(text)
        return tuple(
            (pattern, self.pattern_assemblies[pattern], score)
            for pattern, score in _winning_patterns(
                sequence, self.learned_patterns
            )
        )


def _occurrences(sequence: Sequence[str], pattern: Sequence[str]) -> tuple[int, ...]:
    length = len(pattern)
    return tuple(
        start
        for start in range(len(sequence) - length + 1)
        if tuple(sequence[start : start + length]) == tuple(pattern)
    )


def _winning_patterns(
    sequence: Sequence[str], patterns: dict[tuple[str, ...], float]
) -> tuple[tuple[tuple[str, ...], float], ...]:
    """Global interval competition over internally learned chunk candidates."""
    by_start: dict[int, list[tuple[int, tuple[str, ...], float]]] = defaultdict(list)
    for pattern, score in patterns.items():
        for start in _occurrences(sequence, pattern):
            by_start[start].append((start + len(pattern), pattern, score))

    length = len(sequence)
    best = [0.0] * (length + 1)
    paths: list[tuple[tuple[tuple[str, ...], float], ...]] = [tuple() for _ in range(length + 1)]
    for start in range(length):
        if best[start] >= best[start + 1]:
            best[start + 1] = best[start]
            paths[start + 1] = paths[start]
        for end, pattern, score in by_start.get(start, ()):
            value = best[start] + len(pattern) * math.log1p(score)
            if value > best[end]:
                best[end] = value
                paths[end] = paths[start] + ((pattern, score),)
    return paths[length]


def _sparsify_and_normalize(activity: torch.Tensor, *, max_active: int) -> torch.Tensor:
    if torch.count_nonzero(activity) > max_active:
        winners = torch.topk(activity, max_active).indices
        sparse = torch.zeros_like(activity)
        sparse[winners] = activity[winners]
        activity = sparse
    norm = activity.square().sum().sqrt().clamp_min(1e-8)
    return activity / norm


@dataclass(frozen=True)
class SyllableLocalityAudit:
    boundary_free_syllable_input: bool
    single_connectome_memory_and_output: bool
    local_temporal_synaptic_update: bool
    local_semantic_coactivity_update: bool
    local_output_pre_post_update: bool
    query_control_from_learned_temporal_activity: bool
    no_autograd_or_weight_transport: bool
    global_teacher_target: bool
    global_fact_episode_intersection: bool
    global_window_enumeration: bool
    global_sparse_activity_competition: bool
    global_seed_balancing: bool
    global_output_argmax: bool

    def as_dict(self) -> dict[str, bool]:
        return asdict(self)


class _SyllableDialogueBase:
    def __init__(
        self,
        chunker: FunctionalTemporalChunker | BioLocalTemporalChunker,
        *,
        semantic_units: int = 128,
        semantic_assembly_size: int = 12,
        output_units: int = 64,
        output_assembly_size: int = 4,
        seed: int = 0,
    ):
        self.chunker = chunker
        self.semantic_units = semantic_units
        self.property_bank = SparseAssemblyBank(
            semantic_units, semantic_assembly_size, seed + 101
        )
        self.output_bank = SparseAssemblyBank(
            output_units, output_assembly_size, seed + 202
        )
        self.semantic_cold = torch.zeros(chunker.n_units, semantic_units)
        self.semantic_warm = torch.zeros_like(self.semantic_cold)
        self.bound_units = semantic_units * chunker.n_units
        self.output_cold = torch.zeros(self.bound_units, output_units)
        self.output_warm = torch.zeros_like(self.output_cold)

    @property
    def semantic_weights(self) -> torch.Tensor:
        return self.semantic_cold + self.semantic_warm

    @property
    def output_weights(self) -> torch.Tensor:
        return self.output_cold + self.output_warm

    def observe_streams(self, texts: Iterable[str], *, temporal: bool = True) -> None:
        for text in texts:
            if isinstance(self.chunker, BioLocalTemporalChunker):
                self.chunker.observe(text, plastic=temporal)
            else:
                self.chunker.observe(text)

    def register_syllable_vocabulary(self, texts: Iterable[str]) -> None:
        vocabulary = sorted(
            {
                syllable
                for text in texts
                for syllable in syllable_sequence(text)
            }
        )
        self.chunker.register_syllables(vocabulary)

    def finalize_chunks(self) -> None:
        self.chunker.finalize()

    def register_output_vocabulary(self, tokens: Iterable[str]) -> None:
        for token in tokens:
            self.output_bank.code(token)

    def semantic_state(self, text: str, *, temporal: bool = True) -> torch.Tensor:
        if temporal:
            responses = []
            for pattern, code, score in self.chunker.winning_codes(text):
                response = code @ self.semantic_weights
                strength = float(response.clamp_min(0).square().sum().sqrt())
                if strength > 0:
                    responses.append((strength * math.sqrt(score), response))
            responses.sort(key=lambda item: item[0], reverse=True)
            state = sum(
                (response for _, response in responses[:4]),
                start=torch.zeros(self.semantic_units),
            )
        else:
            feature = self.chunker.feature(text, temporal=False)
            state = feature @ self.semantic_weights
        return _sparsify_and_normalize(state.clamp_min(0), max_active=48)

    def query_control(self, text: str, *, temporal: bool = True) -> torch.Tensor:
        # No intent label: repeated temporal assemblies in the whole utterance
        # constitute the control state.  Semantic and control paths meet only
        # in local conjunctive units below.
        if not temporal:
            return self.chunker.feature(text, temporal=False)
        # Two-to-three-syllable chunks are the word-like control scale in this
        # experiment. Longer recurring phrases remain available to the chunk
        # hierarchy but cannot drown the shared words in a novel paraphrase.
        matched = tuple(
            item for item in self.chunker.matched_codes(text)
            if 2 <= len(item[0]) <= 3
        )
        if not matched:
            return self.chunker.feature(text, temporal=False)
        strengths = [
            float((code @ self.semantic_weights).square().sum().sqrt())
            for _, code, _ in matched
        ]
        maximum = max(strengths, default=0.0)
        # Content chunks acquire strong outgoing semantic synapses during fact
        # learning. Weakly semantic temporal chunks become query control. This
        # role split is learned from connectivity, not supplied as an intent.
        content_patterns = [
            item[0]
            for item, strength in zip(matched, strengths)
            if maximum > 1e-8 and strength > 0.60 * maximum
        ]
        control_chunks = []
        for item, strength in zip(matched, strengths):
            pattern = item[0]
            nested_in_content = any(
                pattern != content
                and _occurrences(content, pattern)
                for content in content_patterns
            )
            if (
                maximum <= 1e-8
                or strength <= 0.60 * maximum
            ) and not nested_in_content:
                control_chunks.append(item)
        if not control_chunks:
            control_chunks = list(matched)
        activity = torch.zeros(self.chunker.n_units)
        for _, code, _ in control_chunks:
            activity.add_(code / code.square().sum().sqrt().clamp_min(1e-8))
        return _sparsify_and_normalize(activity, max_active=192)

    def bound_state(
        self,
        text: str,
        *,
        temporal: bool = True,
        controlled: bool = True,
    ) -> torch.Tensor:
        semantic = self.semantic_state(text, temporal=temporal)
        if controlled:
            control = self.query_control(text, temporal=temporal)
        else:
            control = torch.ones(self.chunker.n_units)
            control /= control.square().sum().sqrt()
        return torch.outer(semantic, control).flatten()

    def answer(
        self,
        text: str,
        *,
        temporal: bool = True,
        controlled: bool = True,
    ) -> tuple[str, float, dict[str, float]]:
        if not self.output_bank.codes:
            raise RuntimeError("output vocabulary is empty")
        state = self.bound_state(
            text, temporal=temporal, controlled=controlled
        )
        output = state @ self.output_weights
        scores = {
            str(token): float(output @ code)
            for token, code in self.output_bank.codes.items()
        }
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
        return ordered[0][0], ordered[0][1] - runner_up, scores

    def consolidate_and_clear(self, *, cycles: int = 100) -> None:
        rate = 1.0 - (1.0 - 0.08) ** cycles
        self.semantic_cold.add_(rate * self.semantic_warm)
        self.output_cold.add_(rate * self.output_warm)
        self.semantic_warm.mul_((1.0 - rate) * 0.9)
        self.output_warm.mul_((1.0 - rate) * 0.9)


class FunctionalSyllableDialogue(_SyllableDialogueBase):
    """Global capability ceiling with learned boundary-free chunks."""

    def __init__(self, *, seed: int = 0):
        super().__init__(FunctionalTemporalChunker(seed=seed), seed=seed)

    def observe_fact(
        self,
        text: str,
        properties: Sequence[str],
        *,
        temporal: bool = True,
        learning_rate: float = 0.20,
    ) -> None:
        target = torch.stack([self.property_bank.code(item) for item in properties]).sum(0)
        sources = (
            [code for _, code, _ in self.chunker.winning_codes(text)]
            if temporal
            else [self.chunker.feature(text, temporal=False)]
        )
        for source in sources:
            normalized = source / source.square().sum().sqrt().clamp_min(1e-8)
            error = target - normalized @ self.semantic_weights
            self.semantic_cold.add_(learning_rate * torch.outer(normalized, error))
        self.semantic_cold.clamp_(-2.0, 2.0)

    def teach_answer(
        self,
        text: str,
        target: str,
        *,
        temporal: bool = True,
        learning_rate: float = 0.20,
    ) -> None:
        code = self.output_bank.code(target)
        state = self.bound_state(text, temporal=temporal)
        self.output_cold.add_(learning_rate * torch.outer(state, code))
        self.output_cold.clamp_(0.0, 4.0)


class BioLocalSyllableDialogue(_SyllableDialogueBase):
    """Boundary-free dialogue using only local internal value updates."""

    def __init__(self, *, seed: int = 0):
        super().__init__(BioLocalTemporalChunker(seed=seed), seed=seed)

    def observe_fact(
        self,
        text: str,
        properties: Sequence[str],
        *,
        temporal: bool = True,
        learning_rate: float = 0.10,
    ) -> None:
        target = torch.stack([self.property_bank.code(item) for item in properties]).sum(0)
        # Each changed edge sees an active presynaptic chunk unit and active
        # postsynaptic property unit; no backward or transposed pathway.
        sources = (
            [code for _, code, _ in self.chunker.winning_codes(text)]
            if temporal
            else [self.chunker.feature(text, temporal=False)]
        )
        for source in sources:
            normalized = source / source.square().sum().sqrt().clamp_min(1e-8)
            local_error = target - normalized @ self.semantic_weights
            self.semantic_warm.add_(
                learning_rate * torch.outer(normalized, local_error)
            )
        self.semantic_warm.clamp_(-2.0, 2.0)

    def teach_answer(
        self,
        text: str,
        target: str,
        *,
        temporal: bool = True,
        learning_rate: float = 0.08,
    ) -> None:
        target_code = self.output_bank.code(target)
        state = self.bound_state(text, temporal=temporal)
        # Teacher clamp is global data delivery, but each synaptic update uses
        # only its own pre activity and postsynaptic target activity.
        self.output_warm.add_(learning_rate * torch.outer(state, target_code))
        self.output_warm.clamp_(0.0, 4.0)

    @staticmethod
    def locality_audit() -> SyllableLocalityAudit:
        return SyllableLocalityAudit(
            boundary_free_syllable_input=True,
            single_connectome_memory_and_output=False,
            local_temporal_synaptic_update=True,
            local_semantic_coactivity_update=True,
            local_output_pre_post_update=True,
            query_control_from_learned_temporal_activity=True,
            no_autograd_or_weight_transport=True,
            global_teacher_target=True,
            global_fact_episode_intersection=False,
            global_window_enumeration=True,
            global_sparse_activity_competition=True,
            global_seed_balancing=True,
            global_output_argmax=True,
        )


class ConnectomeSyllableDialogue:
    """Temporal chunks connected to the existing single I/R/O connectome.

    Supervised fact episodes group several unsegmented utterances about one
    object. Their recurrent common chunk acts as the absent environmental
    object identity; neither its character span nor a word label is supplied.
    """

    def __init__(self, *, seed: int = 0):
        self.chunker = BioLocalTemporalChunker(seed=seed)
        self.connectome = SpatialConnectome(
            ConnectomeConfig(
                n_input=self.chunker.n_units,
                n_substrate=512,
                n_output=64,
                out_degree=192,
                topology="random",
                steps_per_token=2,
                max_region_density=0.18,
                max_output_density=0.10,
                initial_weight=0.08,
                structural_rewire_mode="local_stochastic",
                seed=seed,
            )
        )
        self.concept_patterns: dict[tuple[str, ...], torch.Tensor] = {}
        self.control_generator = torch.Generator().manual_seed(seed + 303)
        self.control_prototypes: list[torch.Tensor] = []
        self.control_gates: list[torch.Tensor] = []
        self.control_gate_usage = torch.zeros(self.connectome.config.n_substrate)

    def register_syllable_vocabulary(self, texts: Iterable[str]) -> None:
        vocabulary = sorted(
            {
                syllable
                for text in texts
                for syllable in syllable_sequence(text)
            }
        )
        self.chunker.register_syllables(vocabulary)

    def register_output_vocabulary(self, tokens: Iterable[str]) -> None:
        self.connectome.register_vocabulary(tokens)

    def observe_streams(self, texts: Iterable[str], *, temporal: bool = True) -> None:
        for text in texts:
            self.chunker.observe(text, plastic=temporal)

    def finalize_chunks(self) -> None:
        self.chunker.finalize()

    def observe_fact_episode(
        self,
        records: Sequence[tuple[str, Sequence[str]]],
        *,
        rounds: int = 40,
        structural_plasticity: bool = True,
    ) -> tuple[str, ...]:
        """Ground the chunk repeated across an unsegmented fact episode."""
        if not records:
            raise ValueError("fact episode must not be empty")
        winners = [
            {pattern for pattern, _, _ in self.chunker.winning_codes(text)}
            for text, _ in records
        ]
        common = set.intersection(*winners)
        if not common:
            raise RuntimeError("fact episode has no recurrent learned chunk")
        # Longest recurrent chunk wins; score resolves equal-length ambiguity.
        pattern = max(
            common,
            key=lambda item: (
                len(item),
                self.chunker.learned_patterns.get(item, 0.0),
            ),
        )
        code = self.chunker.pattern_code(pattern)
        if code is None:
            raise RuntimeError("recurrent fact chunk has no assembly")
        properties = tuple(
            dict.fromkeys(
                property_name
                for _, targets in records
                for property_name in targets
            )
        )
        self.connectome.register_vocabulary(properties)
        self.connectome.learn_sensory_concept(
            code,
            properties,
            rounds=rounds,
            structural_plasticity=structural_plasticity,
        )
        self.concept_patterns[pattern] = code
        return pattern

    def _entity_pattern(self, text: str) -> tuple[str, ...]:
        sequence = syllable_sequence(text)
        present = [
            pattern
            for pattern in self.concept_patterns
            if _occurrences(sequence, pattern)
        ]
        if not present:
            raise RuntimeError("question contains no learned concept chunk")
        return max(present, key=len)

    def _control_pattern(self, text: str, entity: tuple[str, ...]) -> torch.Tensor:
        sequence = syllable_sequence(text)
        entity_spans = [
            (start, start + len(entity))
            for start in _occurrences(sequence, entity)
        ]
        controls = []
        for pattern, code, _ in self.chunker.matched_codes(text):
            if not 2 <= len(pattern) <= 3:
                continue
            pattern_spans = [
                (start, start + len(pattern))
                for start in _occurrences(sequence, pattern)
            ]
            overlaps_entity = any(
                pattern_start < entity_end and entity_start < pattern_end
                for pattern_start, pattern_end in pattern_spans
                for entity_start, entity_end in entity_spans
            )
            if overlaps_entity:
                continue
            if pattern in self.concept_patterns:
                continue
            controls.append(code)
        if not controls:
            return self.chunker.feature(text, temporal=False)
        control = torch.stack(controls).sum(0)
        return _sparsify_and_normalize(control, max_active=192)

    def learn_query_control(
        self,
        text: str,
        *,
        learning_rate: float = 0.20,
        novelty_threshold: float = 0.80,
        gate_fraction: float = 0.125,
    ) -> torch.Tensor:
        """Recruit or update an unlabeled competitive control assembly."""
        entity = self._entity_pattern(text)
        control = self._control_pattern(text, entity)
        similarities = [
            float(torch.dot(control, prototype))
            for prototype in self.control_prototypes
        ]
        if not similarities or max(similarities) < novelty_threshold:
            prototype_index = len(self.control_prototypes)
            self.control_prototypes.append(control.clone())
            count = max(
                1,
                round(
                    self.connectome.config.n_substrate * gate_fraction
                ),
            )
            tie_break = torch.rand(
                len(self.control_gate_usage), generator=self.control_generator
            )
            local = torch.argsort(
                self.control_gate_usage + 1e-3 * tie_break
            )[:count]
            gate = torch.zeros(self.connectome.config.n_units)
            gate[local + self.connectome.config.n_input] = 1.0
            self.control_gates.append(gate)
            self.control_gate_usage[local] += 1.0
        else:
            prototype_index = max(
                range(len(similarities)), key=similarities.__getitem__
            )
            prototype = self.control_prototypes[prototype_index]
            updated = (1.0 - learning_rate) * prototype + learning_rate * control
            self.control_prototypes[prototype_index] = (
                updated / updated.square().sum().sqrt().clamp_min(1e-8)
            )
        return self.control_gates[prototype_index]

    def _learned_control_gate(
        self,
        text: str,
        entity: tuple[str, ...],
    ) -> torch.Tensor:
        control = self._control_pattern(text, entity)
        if not self.control_prototypes:
            raise RuntimeError("query-control assemblies have not been learned")
        similarities = [
            float(torch.dot(control, prototype))
            for prototype in self.control_prototypes
        ]
        winner = max(range(len(similarities)), key=similarities.__getitem__)
        return self.control_gates[winner]

    def bound_state(self, text: str, *, controlled: bool = True) -> torch.Tensor:
        entity = self._entity_pattern(text)
        entity_state = self.connectome.sensory_assembly_state(
            self.concept_patterns[entity], steps=2
        )
        bound = torch.zeros_like(entity_state)
        substrate = self.connectome.region == SUBSTRATE
        if controlled:
            gate = self._learned_control_gate(text, entity)
            bound[substrate] = entity_state[substrate] * gate[substrate]
        else:
            bound[substrate] = entity_state[substrate]
        return bound

    def teach_answer(
        self,
        text: str,
        target: str,
        *,
        structural_plasticity: bool = False,
        learn_control: bool = True,
    ) -> None:
        if learn_control:
            self.learn_query_control(text)
        self.connectome.learn_bound_output(
            self.bound_state(text),
            target,
            structural_plasticity=structural_plasticity,
        )

    def answer(
        self, text: str, *, controlled: bool = True
    ) -> tuple[str, float, dict[str, float]]:
        return self.connectome.predict_bound_output(
            self.bound_state(text, controlled=controlled)
        )

    def consolidate_and_clear(self, *, cycles: int = 100) -> None:
        self.connectome.consolidate(cycles=cycles)
        self.connectome.warm.zero_()
        self.connectome.state.zero_()

    @staticmethod
    def locality_audit() -> SyllableLocalityAudit:
        return SyllableLocalityAudit(
            boundary_free_syllable_input=True,
            single_connectome_memory_and_output=True,
            local_temporal_synaptic_update=True,
            local_semantic_coactivity_update=True,
            local_output_pre_post_update=True,
            query_control_from_learned_temporal_activity=True,
            no_autograd_or_weight_transport=True,
            global_teacher_target=True,
            global_fact_episode_intersection=True,
            global_window_enumeration=True,
            global_sparse_activity_competition=True,
            global_seed_balancing=True,
            global_output_argmax=True,
        )
