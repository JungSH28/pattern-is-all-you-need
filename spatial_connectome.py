"""Spatial sparse connectome prototype.

This is the return path to the original architecture, not a dialogue model yet:

* one sparse directed graph contains input, storage/reasoning, and output units;
* a unit's vector is its outgoing row in that graph;
* tokens/concepts are sparse assemblies, not rows in an embedding table;
* 3-D Euclidean distance biases developmental wiring;
* assembly cosine remains the representational similarity measure;
* hot activity, warm weights, and cold weights share the same substrate.

The first probe intentionally omits conduction delays. Distance has one isolated
causal role here: it changes which synapses exist. Delays should be tested only
after a time scale and a sequence-learning rule are fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


INPUT = 0
SUBSTRATE = 1
OUTPUT = 2


@dataclass(frozen=True)
class ConnectomeConfig:
    n_input: int = 48
    n_substrate: int = 96
    n_output: int = 48
    out_degree: int = 16
    long_range_floor: float = 0.03
    distance_scale: float = 0.22
    topology: str = "distance"
    excitatory_fraction: float = 0.8
    initial_weight: float = 0.18
    threshold: float = 0.08
    leak: float = 0.35
    max_region_density: float = 0.18
    steps_per_token: int = 3
    position_lr: float = 0.08
    position_lr_decay: float = 0.75
    repulsion_radius: float = 0.10
    repulsion_strength: float = 0.03
    consolidation_rate: float = 0.08
    warm_decay: float = 0.90
    recurrent_learning_scale: float = 0.10
    concept_rewire_per_input: int = 4
    output_rewire_per_source: int = 1
    query_gate_fraction: float = 0.50
    seed: int = 0

    @property
    def n_units(self) -> int:
        return self.n_input + self.n_substrate + self.n_output


class SpatialConnectome:
    """A single sparse I/R/O connectome with local state and plastic weights."""

    def __init__(self, config: ConnectomeConfig = ConnectomeConfig()):
        self.config = config
        self.generator = torch.Generator().manual_seed(config.seed)
        self.region = self._make_regions()
        self.positions = self._make_positions()
        self.is_excitatory = self._make_cell_types()
        self.position_plasticity = 1.0
        self.src, self.dst = self._sample_topology()

        # One effective synaptic vector per unit. Warm/cold are time scales of
        # the same edge, not separate computational pathways.
        base = torch.full((len(self.src),), config.initial_weight)
        base *= 0.75 + 0.5 * torch.rand(len(self.src), generator=self.generator)
        self.cold = base
        self.warm = torch.zeros_like(base)
        self.state = torch.zeros(config.n_units)

        self.input_assemblies: dict[str, torch.Tensor] = {}
        self.output_assemblies: dict[str, torch.Tensor] = {}
        self.input_seed_usage = torch.zeros(config.n_input, dtype=torch.long)
        self.output_seed_usage = torch.zeros(config.n_output, dtype=torch.long)
        self.used_input_seed_sets: set[tuple[int, ...]] = set()
        self.used_output_seed_sets: set[tuple[int, ...]] = set()
        self.query_gates: dict[str, torch.Tensor] = {}
        self.query_gate_usage = torch.zeros(config.n_substrate, dtype=torch.long)

    def _make_regions(self) -> torch.Tensor:
        c = self.config
        return torch.cat(
            [
                torch.full((c.n_input,), INPUT),
                torch.full((c.n_substrate,), SUBSTRATE),
                torch.full((c.n_output,), OUTPUT),
            ]
        ).long()

    def _make_positions(self) -> torch.Tensor:
        """Initialize I/R/O as adjacent developmental territories in 3-D."""
        c = self.config
        pos = torch.rand(c.n_units, 3, generator=self.generator)
        pos[: c.n_input, 0] *= 0.25
        r0 = c.n_input
        r1 = r0 + c.n_substrate
        pos[r0:r1, 0] = 0.20 + 0.60 * pos[r0:r1, 0]
        pos[r1:, 0] = 0.75 + 0.25 * pos[r1:, 0]
        return pos

    def _make_cell_types(self) -> torch.Tensor:
        c = self.config
        cell_type = torch.ones(c.n_units, dtype=torch.bool)
        substrate = torch.where(self.region == SUBSTRATE)[0]
        permutation = substrate[torch.randperm(len(substrate), generator=self.generator)]
        n_inhibitory = round(len(substrate) * (1.0 - c.excitatory_fraction))
        cell_type[permutation[:n_inhibitory]] = False
        return cell_type

    def _allowed_destinations(self, source: int) -> torch.Tensor:
        source_region = int(self.region[source])
        if source_region == INPUT:
            allowed = self.region == SUBSTRATE
        elif source_region == SUBSTRATE:
            allowed = (self.region == SUBSTRATE) | (self.region == OUTPUT)
        else:
            # Output-to-substrate feedback lets a nudged output alter the same
            # storage/reasoning substrate during a local target phase.
            allowed = self.region == SUBSTRATE
        allowed[source] = False
        return torch.where(allowed)[0]

    def _sample_topology(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one edge list; distance affects topology, not semantic score."""
        c = self.config
        sources: list[torch.Tensor] = []
        destinations: list[torch.Tensor] = []
        for source in range(c.n_units):
            candidates = self._allowed_destinations(source)
            distance = torch.linalg.vector_norm(
                self.positions[candidates] - self.positions[source], dim=1
            )
            if c.topology == "distance":
                probability = torch.exp(-distance / c.distance_scale) + c.long_range_floor
            elif c.topology == "random":
                probability = torch.ones_like(distance)
            else:
                raise ValueError(f"unknown topology: {c.topology}")
            count = min(c.out_degree, len(candidates))
            selected = torch.multinomial(
                probability, count, replacement=False, generator=self.generator
            )
            sources.append(torch.full((count,), source, dtype=torch.long))
            destinations.append(candidates[selected])
        return torch.cat(sources), torch.cat(destinations)

    @property
    def signs(self) -> torch.Tensor:
        """Dale constraint: all outgoing edges of a unit share one sign."""
        return torch.where(self.is_excitatory[self.src], 1.0, -1.0)

    @property
    def effective_weights(self) -> torch.Tensor:
        return self.signs * (self.cold + self.warm).clamp_min(0.0)

    def outgoing_vector(self, unit: int) -> torch.Tensor:
        """Materialize W[unit, :] for inspection; computation stays sparse."""
        row = torch.zeros(self.config.n_units)
        edge = self.src == unit
        row[self.dst[edge]] = self.effective_weights[edge]
        return row

    def sparse_matrix(self) -> torch.Tensor:
        indices = torch.stack([self.src, self.dst])
        return torch.sparse_coo_tensor(
            indices, self.effective_weights, (self.config.n_units,) * 2
        ).coalesce()

    def register_token(self, token: str, k_input: int = 4, k_output: int = 4) -> None:
        """Assign stable sensory/motor seeds; meaning must form in the substrate."""
        if token in self.input_assemblies:
            return
        c = self.config
        # Independent random draws create accidental seed collisions: later
        # structural learning can then rewrite an unrelated older token. Spread
        # seeds over the least-used units, randomizing only within equal-usage
        # candidates. Semantic overlap must emerge in R rather than be injected
        # as an unstable accident at the sensory/motor boundary.
        inp = self._balanced_unique_seed(
            self.input_seed_usage, k_input, self.used_input_seed_sets
        )
        out = self._balanced_unique_seed(
            self.output_seed_usage, k_output, self.used_output_seed_sets
        )
        self.used_input_seed_sets.add(tuple(sorted(inp.tolist())))
        self.used_output_seed_sets.add(tuple(sorted(out.tolist())))
        self.input_seed_usage[inp] += 1
        self.output_seed_usage[out] += 1
        self.input_assemblies[token] = inp
        self.output_assemblies[token] = out + c.n_input + c.n_substrate

    def _balanced_unique_seed(
        self,
        usage: torch.Tensor,
        count: int,
        used: set[tuple[int, ...]],
    ) -> torch.Tensor:
        """Choose a least-used sparse seed without duplicating an assembly."""
        for _ in range(256):
            tie_break = torch.rand(len(usage), generator=self.generator)
            units = torch.argsort(usage.float() + 1e-3 * tie_break)[:count]
            if tuple(sorted(units.tolist())) not in used:
                return units
        # A duplicate after balanced attempts is unlikely at current sizes;
        # unrestricted random fallback preserves uniqueness near saturation.
        for _ in range(4096):
            units = torch.randperm(len(usage), generator=self.generator)[:count]
            if tuple(sorted(units.tolist())) not in used:
                return units
        raise RuntimeError("sparse token assembly space exhausted")

    def register_vocabulary(self, tokens: Iterable[str]) -> None:
        for token in tokens:
            self.register_token(token)

    def register_query(self, token: str) -> None:
        """Assign a stable R dendritic gate to a query/intention token.

        The gate is a developmental scaffold: the query locally permits or
        suppresses each R unit's semantic activity. Least-used allocation keeps
        early query gates separated; it is explicitly audited as a non-local
        initialization convenience rather than a learned biological rule.
        """
        if token in self.query_gates:
            return
        self.register_token(token)
        c = self.config
        count = max(1, round(c.n_substrate * c.query_gate_fraction))
        tie_break = torch.rand(c.n_substrate, generator=self.generator)
        local = torch.argsort(
            self.query_gate_usage.float() + 1e-3 * tie_break
        )[:count]
        gate = torch.zeros(c.n_units)
        gate[local + c.n_input] = 1.0
        self.query_gate_usage[local] += 1
        self.query_gates[token] = gate

    def sensory_pattern(self, token: str) -> torch.Tensor:
        self.register_token(token)
        pattern = torch.zeros(self.config.n_units)
        pattern[self.input_assemblies[token]] = 1.0
        return pattern

    def output_pattern(self, token: str) -> torch.Tensor:
        self.register_token(token)
        pattern = torch.zeros(self.config.n_units)
        pattern[self.output_assemblies[token]] = 1.0
        return pattern

    def simultaneous_state(
        self, tokens: Sequence[str], *, steps: int = 2
    ) -> torch.Tensor:
        """Evoked state for co-present sensory properties.

        This represents observing an entity and several properties together,
        rather than a temporal language sequence. It is the target phase used
        to organize an entity assembly before any category label is supplied.
        """
        sensory = torch.zeros(self.config.n_units)
        for token in tokens:
            sensory = torch.maximum(sensory, self.sensory_pattern(token))
        state = torch.zeros_like(sensory)
        for _ in range(steps):
            state = self.step(state, sensory=sensory)
        return state

    @staticmethod
    def assembly_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
        """Representational similarity; deliberately independent of 3-D distance."""
        return float(F.cosine_similarity(left[None], right[None]).item())

    def _cap_region_activity(self, activity: torch.Tensor) -> torch.Tensor:
        capped = activity.clone()
        for region_id in (INPUT, SUBSTRATE, OUTPUT):
            indices = torch.where(self.region == region_id)[0]
            limit = max(1, round(len(indices) * self.config.max_region_density))
            values = capped[indices]
            if torch.count_nonzero(values) > limit:
                keep = torch.topk(values, limit).indices
                mask = torch.zeros_like(values, dtype=torch.bool)
                mask[keep] = True
                capped[indices[~mask]] = 0.0
        return capped

    def step(
        self,
        state: torch.Tensor,
        sensory: torch.Tensor | None = None,
        teacher: torch.Tensor | None = None,
    ) -> torch.Tensor:
        drive = torch.zeros_like(state)
        drive.scatter_add_(0, self.dst, state[self.src] * self.effective_weights)
        activity = F.relu((1.0 - self.config.leak) * state + drive - self.config.threshold)
        activity = self._cap_region_activity(activity)
        if sensory is not None:
            input_units = self.region == INPUT
            activity[input_units] = torch.maximum(activity[input_units], sensory[input_units])
        if teacher is not None:
            output_units = self.region == OUTPUT
            activity[output_units] = torch.maximum(activity[output_units], teacher[output_units])
        maximum = activity.max()
        if maximum > 1.0:
            activity = activity / maximum
        return activity

    def run_sequence(
        self,
        tokens: Sequence[str],
        *,
        reset: bool = True,
        keep_trace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        state = torch.zeros(self.config.n_units) if reset else self.state.clone()
        trace: list[torch.Tensor] = []
        for token in tokens:
            sensory = self.sensory_pattern(token)
            for inner in range(self.config.steps_per_token):
                state = self.step(state, sensory=sensory if inner == 0 else None)
                trace.append(state.clone())
        self.state = state
        return (state, trace) if keep_trace else state

    def output_scores(self, state: torch.Tensor, tokens: Sequence[str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for token in tokens:
            pattern = self.output_pattern(token)
            scores[token] = float((state * pattern).sum().item())
        return scores

    def predict_output(
        self,
        context: Sequence[str],
        candidates: Sequence[str],
        *,
        readout_steps: int = 1,
    ) -> tuple[str, float, dict[str, float]]:
        """Evoke and decode an output assembly after a sensory context.

        The extra silent step is a physical R->O propagation step, not an
        external similarity head. Scores read activity on each candidate's O
        assembly; the highest-scoring token is the emitted symbol.
        """
        if not candidates:
            raise ValueError("candidates must not be empty")
        state = self.run_sequence(context, reset=True)
        assert isinstance(state, torch.Tensor)
        for _ in range(readout_steps):
            state = self.step(state)
        self.state = state
        scores = self.output_scores(state, candidates)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
        return ordered[0][0], ordered[0][1] - runner_up, scores

    def query_bound_state(
        self, query: str, entity: str, *, gated: bool = True
    ) -> torch.Tensor:
        """Bind an entity assembly to a query through local dendritic gating."""
        self.register_query(query)
        entity_state = self.simultaneous_state((entity,), steps=2)
        if not gated:
            return entity_state
        bound = torch.zeros_like(entity_state)
        substrate = self.region == SUBSTRATE
        bound[substrate] = (
            entity_state[substrate] * self.query_gates[query][substrate]
        )
        return bound

    def predict_query_output(
        self,
        query: str,
        entity: str,
        *,
        gated: bool = True,
        readout_steps: int = 1,
    ) -> tuple[str, float, dict[str, float]]:
        """Answer from the entire registered output vocabulary.

        No per-question candidate list is accepted. The query gates an entity
        state, silent R->O propagation evokes motor assemblies, and every
        registered token competes in the same decoder.
        """
        state = self.query_bound_state(query, entity, gated=gated)
        for _ in range(readout_steps):
            state = self.step(state)
        self.state = state
        candidates = tuple(self.output_assemblies)
        scores = self.output_scores(state, candidates)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
        return ordered[0][0], ordered[0][1] - runner_up, scores

    def learn_output_association(
        self,
        context: Sequence[str],
        target: str,
        *,
        learning_rate: float = 0.03,
        structural_plasticity: bool = False,
        readout_steps: int = 1,
    ) -> float:
        """Link an evoked R assembly to a language-output assembly locally.

        A held presynaptic R trace and the target O neuron's local difference
        between teacher activity and its own evoked activity determine the
        update. Optional structural plasticity gives active excitatory R units
        sparse access to the taught O assembly at fixed out-degree.
        """
        free = self.run_sequence(context, reset=True)
        assert isinstance(free, torch.Tensor)
        target_pattern = self.output_pattern(target)
        if structural_plasticity:
            self._rewire_output_inputs(free, target_pattern)

        readout = free
        for _ in range(readout_steps):
            readout = self.step(readout)
        output_edge = (self.region[self.src] == SUBSTRATE) & (
            self.region[self.dst] == OUTPUT
        )
        output_error = target_pattern[self.dst[output_edge]] - readout[
            self.dst[output_edge]
        ]
        delta = (
            self.signs[output_edge]
            * free[self.src[output_edge]]
            * output_error
        )
        self.warm[output_edge] = (
            self.warm[output_edge] + learning_rate * delta
        ).clamp(min=-1.5, max=1.5)
        return float(delta.abs().mean().item())

    def learn_query_association(
        self,
        query: str,
        entity: str,
        target: str,
        *,
        learning_rate: float = 0.03,
        structural_plasticity: bool = False,
    ) -> float:
        """Teach a query-conditioned answer with the local R->O rule."""
        free = self.query_bound_state(query, entity, gated=True)
        target_pattern = self.output_pattern(target)
        if structural_plasticity:
            self._rewire_output_inputs(free, target_pattern)
        readout = self.step(free)
        output_edge = (self.region[self.src] == SUBSTRATE) & (
            self.region[self.dst] == OUTPUT
        )
        output_error = target_pattern[self.dst[output_edge]] - readout[
            self.dst[output_edge]
        ]
        delta = (
            self.signs[output_edge]
            * free[self.src[output_edge]]
            * output_error
        )
        self.warm[output_edge] = (
            self.warm[output_edge] + learning_rate * delta
        ).clamp(min=-1.5, max=1.5)
        return float(delta.abs().mean().item())

    def _rewire_output_inputs(
        self, concept_state: torch.Tensor, target_pattern: torch.Tensor
    ) -> None:
        """Create sparse R->O access from a concept to its spoken token.

        Each active excitatory R source keeps at most the configured number of
        direct synapses into this target assembly. Its weakest other R->O edge
        is replaced, so total degree is unchanged and consolidated strong edges
        resist later replacement. A new synapse starts in warm memory.
        """
        required = self.config.output_rewire_per_source
        if required <= 0:
            return
        active_sources = torch.where(
            (self.region == SUBSTRATE)
            & self.is_excitatory
            & (concept_state > 0)
        )[0]
        target_units = torch.where((self.region == OUTPUT) & (target_pattern > 0))[0]
        for source in active_sources.tolist():
            output_edges = torch.where(
                (self.src == source) & (self.region[self.dst] == OUTPUT)
            )[0]
            if len(output_edges) == 0:
                continue
            existing = self.dst[output_edges]
            present = int(torch.isin(existing, target_units).sum().item())
            number = min(required - present, len(output_edges))
            if number <= 0:
                continue
            candidates = target_units[~torch.isin(target_units, existing)]
            number = min(number, len(candidates))
            if number <= 0:
                continue

            selected = torch.randperm(
                len(candidates), generator=self.generator
            )[:number]
            replacement_pool = output_edges[~torch.isin(existing, target_units)]
            magnitude = (self.cold + self.warm).abs()[replacement_pool]
            replacement = replacement_pool[
                torch.topk(magnitude, number, largest=False).indices
            ]
            self.dst[replacement] = candidates[selected]
            self.cold[replacement] = 0.0
            self.warm[replacement] = self.config.initial_weight

    def learn_association(
        self,
        context: Sequence[str],
        target: str,
        *,
        learning_rate: float = 0.08,
        target_steps: int = 3,
    ) -> float:
        """One contrastive local update on the single edge list.

        The free and output-nudged phases provide a local correlation difference
        on every existing synapse. No transposed forward matrix or autograd is
        used. This is a probe rule, not a claim about the brain's final rule.
        """
        free = self.run_sequence(context, reset=True)
        assert isinstance(free, torch.Tensor)
        target_pattern = self.output_pattern(target)
        nudged = free.clone()
        for _ in range(target_steps):
            nudged = self.step(nudged, teacher=target_pattern)

        free_corr = free[self.src] * free[self.dst]
        nudged_corr = nudged[self.src] * nudged[self.dst]
        delta = self.signs * (nudged_corr - free_corr)

        # A target output supplies a local postsynaptic error. R->O synapses
        # combine it with their own presynaptic context activity; this is the
        # minimal branch-specific rule needed for AB->C and DB->E. Recurrent
        # edges still receive the weaker contrastive update above.
        output_edge = self.region[self.dst] == OUTPUT
        output_error = target_pattern[self.dst] - free[self.dst]
        delta[output_edge] = (
            self.signs[output_edge]
            * free[self.src[output_edge]]
            * output_error[output_edge]
        )
        recurrent_edge = (self.region[self.src] == SUBSTRATE) & (
            self.region[self.dst] == SUBSTRATE
        )
        delta[recurrent_edge] *= self.config.recurrent_learning_scale
        # Warm plasticity is signed: positive change is LTP and negative change
        # is LTD. Dale's law constrains the sign of the effective outgoing
        # synapse, while its non-negative magnitude may strengthen or weaken.
        self.warm.add_(learning_rate * delta).clamp_(min=-1.5, max=1.5)
        return float(delta.abs().mean().item())

    def learn_concept(
        self,
        entity: str,
        properties: Sequence[str],
        *,
        rounds: int = 40,
        learning_rate: float = 0.30,
        settle_steps: int = 2,
        consolidate_every: int = 10,
    ) -> None:
        """Organize an entity assembly from co-present property assemblies.

        The entity receives no category label. Its active sensory units provide
        the presynaptic factor; each existing I->R synapse sees only the local
        difference between the property's target-phase activity and the
        entity-only free phase at its postsynaptic R unit. This is a local
        delta/contrastive rule on the same connectome edge list.
        """
        entity_sensory = self.sensory_pattern(entity)
        entity_edge = (
            (self.region[self.src] == INPUT)
            & (self.region[self.dst] == SUBSTRATE)
            & (entity_sensory[self.src] > 0)
        )
        target = self.simultaneous_state(properties, steps=settle_steps)
        self._rewire_concept_inputs(entity_sensory, target)
        # Recompute after structural plasticity changed destinations.
        entity_edge = (
            (self.region[self.src] == INPUT)
            & (self.region[self.dst] == SUBSTRATE)
            & (entity_sensory[self.src] > 0)
        )
        for iteration in range(rounds):
            free = self.simultaneous_state((entity,), steps=settle_steps)
            target = self.simultaneous_state(properties, steps=settle_steps)
            delta = torch.zeros_like(self.warm)
            delta[entity_edge] = (
                target[self.dst[entity_edge]] - free[self.dst[entity_edge]]
            )
            self.warm.add_(learning_rate * delta).clamp_(min=-1.5, max=1.5)
            if iteration % consolidate_every == 0:
                self.consolidate()

    def _rewire_concept_inputs(
        self, entity_sensory: torch.Tensor, target: torch.Tensor
    ) -> None:
        """Form a few coactivity-driven I->R synapses at fixed out-degree.

        Each active entity input unit replaces existing edges whose targets are
        least active in the property phase. New targets must be active R units;
        distance supplies a wiring prior only in spatial topology conditions.
        The operation preserves the number of edges per source and never uses a
        category label.
        """
        count = self.config.concept_rewire_per_input
        if count <= 0:
            return
        active_inputs = torch.where(
            (self.region == INPUT) & (entity_sensory > 0)
        )[0]
        active_targets = torch.where(
            (self.region == SUBSTRATE) & (target > 0)
        )[0]
        for source in active_inputs.tolist():
            source_edges = torch.where(
                (self.src == source) & (self.region[self.dst] == SUBSTRATE)
            )[0]
            existing = self.dst[source_edges]
            available_mask = ~torch.isin(active_targets, existing)
            candidates = active_targets[available_mask]
            number = min(count, len(candidates), len(source_edges))
            if number == 0:
                continue

            probability = target[candidates].clamp_min(1e-6)
            if self.config.topology == "distance":
                distance = torch.linalg.vector_norm(
                    self.positions[candidates] - self.positions[source], dim=1
                )
                probability = probability * (
                    torch.exp(-distance / self.config.distance_scale)
                    + self.config.long_range_floor
                )
            selected = torch.multinomial(
                probability,
                number,
                replacement=False,
                generator=self.generator,
            )
            new_destinations = candidates[selected]

            # Replace edges least useful for the observed property assembly.
            replacement = torch.topk(
                target[existing], number, largest=False
            ).indices
            edge_indices = source_edges[replacement]
            self.dst[edge_indices] = new_destinations
            self.cold[edge_indices] = self.config.initial_weight
            self.warm[edge_indices] = 0.0

    def consolidate(self, cycles: int = 1) -> None:
        """Transfer fast synaptic change into the slow variable on each edge."""
        if cycles < 0:
            raise ValueError("cycles must be non-negative")
        rate = self.config.consolidation_rate
        for _ in range(cycles):
            self.cold.add_(rate * self.warm).clamp_(min=0.0, max=2.0)
            self.warm.mul_(self.config.warm_decay)

    def develop_positions(self, activity_samples: torch.Tensor, epochs: int = 1) -> None:
        """Early activity-dependent movement followed by topology resampling.

        Attraction is restricted to already connected, co-active pairs. A short
        range density force prevents spatial collapse. The learning rate decays,
        representing large early and small late positional plasticity.
        Call this before task learning because resampling replaces the edge set.
        """
        if activity_samples.ndim != 2 or activity_samples.shape[1] != self.config.n_units:
            raise ValueError("activity_samples must have shape (samples, n_units)")

        c = self.config
        for _ in range(epochs):
            force = torch.zeros_like(self.positions)
            for activity in activity_samples:
                coactivity = activity[self.src] * activity[self.dst]
                displacement = self.positions[self.dst] - self.positions[self.src]
                edge_force = coactivity[:, None] * displacement
                force.index_add_(0, self.src, edge_force)
                force.index_add_(0, self.dst, -edge_force)

            difference = self.positions[:, None, :] - self.positions[None, :, :]
            distance = torch.linalg.vector_norm(difference, dim=-1).clamp_min(1e-4)
            nearby = (distance < c.repulsion_radius) & (distance > 1e-3)
            repulsion = (difference / distance[..., None].pow(2)) * nearby[..., None]
            force += c.repulsion_strength * repulsion.sum(dim=1)
            force /= max(1, len(activity_samples))

            step_size = c.position_lr * self.position_plasticity
            self.positions.add_(step_size * force).clamp_(0.0, 1.0)
            self.position_plasticity *= c.position_lr_decay

        self.src, self.dst = self._sample_topology()
        base = torch.full((len(self.src),), c.initial_weight)
        self.cold = base
        self.warm = torch.zeros_like(base)

    def mean_edge_distance(self) -> float:
        distance = torch.linalg.vector_norm(
            self.positions[self.src] - self.positions[self.dst], dim=1
        )
        return float(distance.mean().item())

    def mean_allowed_distance(self) -> float:
        """Mean distance of all anatomically allowed pairs for topology ablation."""
        distances = []
        for source in range(self.config.n_units):
            destination = self._allowed_destinations(source)
            distances.append(
                torch.linalg.vector_norm(
                    self.positions[destination] - self.positions[source], dim=1
                )
            )
        return float(torch.cat(distances).mean().item())


def structural_probe() -> dict[str, float]:
    """Small deterministic smoke probe used by README-level development."""
    model = SpatialConnectome()
    model.register_vocabulary(["cat", "dog", "animal", "bank", "river", "money"])
    cat_dog = model.assembly_cosine(
        model.sensory_pattern("cat"), model.sensory_pattern("dog")
    )
    bank_river = model.run_sequence(["river", "bank"])
    bank_money = model.run_sequence(["money", "bank"])
    assert isinstance(bank_river, torch.Tensor)
    assert isinstance(bank_money, torch.Tensor)
    context_difference = 1.0 - model.assembly_cosine(bank_river, bank_money)
    return {
        "mean_edge_distance": model.mean_edge_distance(),
        "mean_allowed_distance": model.mean_allowed_distance(),
        "random_seed_cosine": cat_dog,
        "context_state_difference": context_difference,
        "density": float((bank_river > 0).float().mean().item()),
    }


if __name__ == "__main__":
    for key, value in structural_probe().items():
        print(f"{key}: {value:.6f}")
