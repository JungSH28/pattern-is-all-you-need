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
    max_output_density: float | None = None
    # Homeostatic firing replaces the global top-k/max activity selection. Each
    # neuron owns a threshold and adjusts it from its own firing rate alone, so
    # the active count varies per input instead of being ranked into a quota.
    homeostatic_threshold: bool = False
    firing_rate_floor: float = 0.02
    firing_rate_ceiling: float = 0.20
    # The rate estimate is the sensor and the threshold is the actuator, so
    # the actuator must not outrun it: 0.02 overshoots into silence, 0.0002
    # never controls and the region saturates.
    threshold_adaptation_rate: float = 0.002
    firing_rate_trace: float = 0.002
    minimum_firing_threshold: float = 0.01
    intrinsic_stability_strength: float = 0.0
    intrinsic_stability_rate: float = 1.0
    steps_per_token: int = 3
    position_lr: float = 0.08
    position_lr_decay: float = 0.75
    repulsion_radius: float = 0.10
    repulsion_strength: float = 0.03
    consolidation_rate: float = 0.08
    warm_decay: float = 0.90
    synaptic_stability_rate: float = 1.0
    synaptic_stability_strength: float = 0.0
    target_local_output_plasticity: bool = False
    output_commitment_threshold: float = 0.0
    output_synaptic_scaling: bool = False
    output_feedback_learning_scale: float = 0.0
    concept_rewire_stability_threshold: float = 0.0
    dendritic_output_enabled: bool = False
    dendritic_branch_novelty_threshold: float = 0.65
    max_dendritic_branches_per_output: int = 16
    dendritic_branch_initial_warm: float = 1.0
    recurrent_learning_scale: float = 0.10
    concept_rewire_per_input: int = 4
    output_rewire_per_source: int = 1
    structural_rewire_mode: str = "weakest"
    query_gate_fraction: float = 0.50
    seed: int = 0

    @property
    def n_units(self) -> int:
        return self.n_input + self.n_substrate + self.n_output


@dataclass
class OutputDendriticBranch:
    """One O-neuron-local coincidence compartment and its time scales."""

    sources: torch.Tensor
    weights: torch.Tensor
    cold: float = 0.0
    warm: float = 1.0


def _product(pair: tuple[float, float]) -> float:
    return pair[0] * pair[1]


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
        # Edge-local metaplastic state. Only actual warm->cold transfer raises
        # stability, so untouched developmental weights remain plastic.
        self.stability = torch.zeros_like(base)
        self.stability_tag = torch.zeros_like(base)
        self.state = torch.zeros(config.n_units)
        # Intrinsic excitability, one value per neuron (Turrigiano homeostasis).
        self.firing_threshold = torch.full((config.n_units,), config.threshold)
        self.firing_rate = torch.full(
            (config.n_units,),
            0.5 * (config.firing_rate_floor + config.firing_rate_ceiling),
        )
        self.threshold_stability = torch.zeros(config.n_units)
        # How much this neuron has recently retuned; drives its own locking.
        self.threshold_change = torch.zeros(config.n_units)

        self.input_assemblies: dict[str, torch.Tensor] = {}
        self.output_assemblies: dict[str, torch.Tensor] = {}
        self.input_seed_usage = torch.zeros(config.n_input, dtype=torch.long)
        self.output_seed_usage = torch.zeros(config.n_output, dtype=torch.long)
        self.used_input_seed_sets: set[tuple[int, ...]] = set()
        self.used_output_seed_sets: set[tuple[int, ...]] = set()
        self.query_gates: dict[str, torch.Tensor] = {}
        self.query_gate_usage = torch.zeros(config.n_substrate, dtype=torch.long)
        self.output_dendritic_branches: dict[
            int, list[OutputDendriticBranch]
        ] = {}

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

    @property
    def local_plasticity(self) -> torch.Tensor:
        """Per-edge learning availability from that edge's own history."""
        strength = self.config.synaptic_stability_strength
        return 1.0 / (1.0 + strength * self.stability)

    def _output_error(
        self,
        target_pattern: torch.Tensor,
        readout: torch.Tensor,
        output_edge: torch.Tensor,
    ) -> torch.Tensor:
        """Postsynaptic teaching signal available at each R->O contact."""
        target = target_pattern[self.dst[output_edge]]
        error = target - readout[self.dst[output_edge]]
        if self.config.target_local_output_plasticity:
            # A clamped target neuron can modify its own incoming contacts;
            # silent non-target neurons receive no exact negative-label signal.
            error = error * (target > 0)
        return error

    def _tag_target_output_edges(
        self,
        output_edge: torch.Tensor,
        presynaptic_state: torch.Tensor,
        target_pattern: torch.Tensor,
    ) -> None:
        """Set a local capture tag only where target clamp meets active pre."""
        indices = torch.where(output_edge)[0]
        tagged = (
            (presynaptic_state[self.src[indices]] > 0)
            & (target_pattern[self.dst[indices]] > 0)
        )
        self.stability_tag[indices[tagged]] = 1.0

    def _output_source_plasticity(
        self,
        target_pattern: torch.Tensor,
        output_edge: torch.Tensor,
    ) -> torch.Tensor:
        """Gate target learning by each R neuron's own consolidated terminals.

        A source with a mature R->O contact may keep learning that target but
        does not acquire a competing target. No other source or network-wide
        importance estimate is consulted.
        """
        threshold = self.config.output_commitment_threshold
        if threshold <= 0.0:
            return torch.ones(
                int(output_edge.sum().item()), dtype=self.warm.dtype
            )
        target_units = target_pattern > 0
        mature = output_edge & (self.stability >= threshold)
        committed = torch.zeros(self.config.n_units, dtype=torch.bool)
        same_target = torch.zeros_like(committed)
        committed[self.src[mature]] = True
        matching = mature & target_units[self.dst]
        same_target[self.src[matching]] = True
        source_allowed = ~committed | same_target
        return source_allowed[self.src[output_edge]].float()

    def _learn_output_feedback(
        self,
        presynaptic_state: torch.Tensor,
        target_pattern: torch.Tensor,
        learning_rate: float,
    ) -> None:
        """Bind a clamped O assembly back to its active R state locally."""
        scale = self.config.output_feedback_learning_scale
        if scale <= 0.0:
            return
        edge = (
            (self.region[self.src] == OUTPUT)
            & (self.region[self.dst] == SUBSTRATE)
            & (target_pattern[self.src] > 0)
        )
        delta = (
            target_pattern[self.src[edge]]
            * presynaptic_state[self.dst[edge]]
            * self.local_plasticity[edge]
        )
        self.warm[edge] = (
            self.warm[edge] + learning_rate * scale * delta
        ).clamp(min=-1.5, max=1.5)
        self.stability_tag[edge & (presynaptic_state[self.dst] > 0)] = 1.0

    def _normalized_branch_activity(
        self, state: torch.Tensor, *, with_output_context: bool = False
    ) -> torch.Tensor:
        activity = torch.zeros_like(state)
        sources = self.region == SUBSTRATE
        if with_output_context:
            # Recently emitted O assemblies are ordinary presynaptic partners
            # for an O neuron's branches, giving serial order without an
            # external decoder.
            sources = sources | (self.region == OUTPUT)
        activity[sources] = state[sources]
        return activity / activity.square().sum().sqrt().clamp_min(1e-8)

    @staticmethod
    def _branch_strength(branch: OutputDendriticBranch) -> float:
        return min(1.0, max(0.0, branch.cold + branch.warm))

    def _branch_response(
        self, normalized: torch.Tensor, branch: OutputDendriticBranch
    ) -> float:
        return float(torch.dot(normalized[branch.sources], branch.weights))

    def learn_dendritic_output(
        self,
        bound_state: torch.Tensor,
        target_pattern: torch.Tensor,
        *,
        with_output_context: bool = False,
    ) -> float:
        """Recruit O-local branches from target clamp and coincident R activity."""
        normalized = self._normalized_branch_activity(
            bound_state, with_output_context=with_output_context
        )
        sources = torch.where(normalized > 0)[0]
        if len(sources) == 0:
            return 0.0
        weights = normalized[sources].clone()
        recruited = 0
        target_units = torch.where(
            (self.region == OUTPUT) & (target_pattern > 0)
        )[0]
        for destination in target_units.tolist():
            branches = self.output_dendritic_branches.setdefault(destination, [])
            responses = [
                self._branch_response(normalized, branch) for branch in branches
            ]
            if responses and max(responses) >= (
                self.config.dendritic_branch_novelty_threshold
            ):
                continue
            new_branch = OutputDendriticBranch(
                sources=sources.clone(),
                weights=weights.clone(),
                warm=self.config.dendritic_branch_initial_warm,
            )
            if len(branches) < self.config.max_dendritic_branches_per_output:
                branches.append(new_branch)
                recruited += 1
                continue
            immature = [
                index for index, branch in enumerate(branches)
                if branch.cold < 0.05
            ]
            if immature:
                replace = min(immature, key=responses.__getitem__)
                branches[replace] = new_branch
                recruited += 1
        return recruited / max(1, len(target_units))

    def dendritic_output_state(
        self, bound_state: torch.Tensor, *, with_output_context: bool = False
    ) -> torch.Tensor:
        """Let each O neuron emit its strongest local branch coincidence."""
        normalized = self._normalized_branch_activity(
            bound_state, with_output_context=with_output_context
        )
        state = bound_state.clone()
        state[self.region == OUTPUT] = 0.0
        for destination, branches in self.output_dendritic_branches.items():
            state[destination] = max(
                (
                    self._branch_response(normalized, branch)
                    * self._branch_strength(branch)
                    for branch in branches
                ),
                default=0.0,
            )
        return state

    def _region_unit(self, state: torch.Tensor, region: int) -> torch.Tensor:
        activity = torch.zeros_like(state)
        selected = self.region == region
        activity[selected] = state[selected]
        norm = activity.square().sum().sqrt()
        return activity / norm if norm > 1e-8 else activity

    def _sequence_branch_response(
        self, state: torch.Tensor, branch: OutputDendriticBranch
    ) -> tuple[float, float]:
        """Coincidence of two separately normalized streams.

        A branch's R sources carry what is meant and its O sources carry what
        was just said.  Concatenating them into one normalized vector makes the
        two compete for the same budget: raising the emitted trace enough to end
        a sentence starves the query gate that selects the content.  Requiring
        both matches instead keeps the streams independent, and a branch
        recruited with an empty trace stops firing once the trace fills, which
        ends the sentence without a terminal symbol.
        """
        regions = self.region[branch.sources]
        matches = []
        for region in (SUBSTRATE, OUTPUT):
            selected = regions == region
            weights = branch.weights[selected]
            present = float(state[self.region == region].abs().sum()) > 1e-8
            if not len(weights):
                # This branch expects silence in the region; any activity there
                # means the context is not the one it was recruited for.
                matches.append(0.0 if present else 1.0)
                continue
            if not present:
                matches.append(0.0)
                continue
            norm = weights.square().sum().sqrt().clamp_min(1e-8)
            normalized = self._region_unit(state, region)
            matches.append(
                float(torch.dot(normalized[branch.sources[selected]], weights / norm))
            )
        return matches[0], matches[1]

    def dendritic_sequence_emission(
        self, bound_state: torch.Tensor
    ) -> dict[str, tuple[float, float]]:
        """Per registered token, its best branch ``(score, match)``.

        ``score`` weights the full coincidence by the branch's consolidated
        strength and decides which token wins.  The reported match is the order
        stream alone.  A held-out entity weakens the meaning stream while its
        order stream stays intact, whereas a finished sentence leaves no branch
        expecting the present trace, so thresholding the order stream separates
        "say the next syllable" from "stop" where the product cannot.
        """
        emission: dict[str, tuple[float, float]] = {}
        for token, units in self.output_assemblies.items():
            score = 0.0
            match_total = 0.0
            for unit in units.tolist():
                best_score = 0.0
                best_match = 0.0
                for branch in self.output_dendritic_branches.get(unit, ()):
                    meaning, order = self._sequence_branch_response(
                        bound_state, branch
                    )
                    unit_score = meaning * order * self._branch_strength(branch)
                    if unit_score > best_score:
                        best_score, best_match = unit_score, order
                # The whole assembly competes: evidence sums over its units,
                # exactly as the single-token readout does.
                score += best_score
                match_total += best_match
            emission[str(token)] = (score, match_total / max(1, len(units)))
        return emission

    def learn_dendritic_sequence(
        self, bound_state: torch.Tensor, target_pattern: torch.Tensor
    ) -> int:
        """Recruit a two-stream branch where the target is not already produced."""
        normalized = self._normalized_branch_activity(
            bound_state, with_output_context=True
        )
        sources = torch.where(normalized != 0)[0]
        if len(sources) == 0:
            return 0
        recruited = 0
        targets = torch.where((self.region == OUTPUT) & (target_pattern > 0))[0]
        for destination in targets.tolist():
            branches = self.output_dendritic_branches.setdefault(destination, [])
            responses = [
                _product(self._sequence_branch_response(bound_state, branch))
                for branch in branches
            ]
            if responses and max(responses) >= (
                self.config.dendritic_branch_novelty_threshold
            ):
                continue
            new_branch = OutputDendriticBranch(
                sources=sources.clone(),
                weights=normalized[sources].clone(),
                warm=self.config.dendritic_branch_initial_warm,
            )
            if len(branches) < self.config.max_dendritic_branches_per_output:
                branches.append(new_branch)
                recruited += 1
                continue
            immature = [
                index for index, branch in enumerate(branches) if branch.cold < 0.05
            ]
            if immature:
                branches[min(immature, key=responses.__getitem__)] = new_branch
                recruited += 1
        return recruited

    def dendritic_branch_count(self) -> int:
        return sum(map(len, self.output_dendritic_branches.values()))

    def clear_fast_synapses(self) -> None:
        """Erase warm linear and dendritic state while retaining cold memory."""
        self.warm.zero_()
        for branches in self.output_dendritic_branches.values():
            for branch in branches:
                branch.warm = 0.0

    def _dendritic_contacts(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sources = []
        destinations = []
        values = []
        for destination, branches in self.output_dendritic_branches.items():
            for branch in branches:
                strength = self._branch_strength(branch)
                sources.append(branch.sources)
                destinations.append(
                    torch.full_like(branch.sources, destination)
                )
                sign = torch.where(
                    self.is_excitatory[branch.sources], 1.0, -1.0
                )
                values.append(sign * strength * branch.weights)
        if not sources:
            empty_index = torch.empty(0, dtype=torch.long)
            return empty_index, empty_index.clone(), torch.empty(0)
        return torch.cat(sources), torch.cat(destinations), torch.cat(values)

    def outgoing_vector(self, unit: int) -> torch.Tensor:
        """Materialize W[unit, :] for inspection; computation stays sparse."""
        row = torch.zeros(self.config.n_units)
        edge = self.src == unit
        row[self.dst[edge]] = self.effective_weights[edge]
        branch_src, branch_dst, branch_weight = self._dendritic_contacts()
        branch_edge = branch_src == unit
        row.index_add_(0, branch_dst[branch_edge], branch_weight[branch_edge])
        return row

    def sparse_matrix(self) -> torch.Tensor:
        branch_src, branch_dst, branch_weight = self._dendritic_contacts()
        indices = torch.stack(
            [torch.cat((self.src, branch_src)), torch.cat((self.dst, branch_dst))]
        )
        values = torch.cat((self.effective_weights, branch_weight))
        return torch.sparse_coo_tensor(
            indices, values, (self.config.n_units,) * 2
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

    def expanded_input_pattern(self, pattern: torch.Tensor) -> torch.Tensor:
        """Place an externally formed sensory assembly on the shared I region."""
        if pattern.ndim != 1 or len(pattern) != self.config.n_input:
            raise ValueError("input pattern must have shape (n_input,)")
        expanded = torch.zeros(self.config.n_units)
        expanded[: self.config.n_input] = pattern
        return expanded

    def sensory_assembly_state(
        self, pattern: torch.Tensor, *, steps: int = 2, plastic: bool = False
    ) -> torch.Tensor:
        """Evoke R activity from a learned input assembly without a token ID."""
        sensory = self.expanded_input_pattern(pattern)
        state = torch.zeros(self.config.n_units)
        for inner in range(steps):
            state = self.step(
                state,
                sensory=sensory if inner == 0 else None,
                plastic=plastic,
            )
        return state

    def sensory_assembly_gate(
        self, pattern: torch.Tensor, *, fraction: float = 0.50
    ) -> torch.Tensor:
        """Project a control assembly through its physical I->R synapses.

        The gate is selected from postsynaptic drive, not from an external
        query label. Global top-k remains an audited activity scaffold.
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        sensory = self.expanded_input_pattern(pattern)
        edge = (self.region[self.src] == INPUT) & (
            self.region[self.dst] == SUBSTRATE
        )
        drive = torch.zeros(self.config.n_units)
        drive.scatter_add_(
            0,
            self.dst[edge],
            sensory[self.src[edge]] * self.effective_weights[edge].abs(),
        )
        substrate = torch.where(self.region == SUBSTRATE)[0]
        count = max(1, round(len(substrate) * fraction))
        winners = substrate[torch.topk(drive[substrate], count).indices]
        gate = torch.zeros(self.config.n_units)
        gate[winners] = 1.0
        return gate

    def output_pattern(self, token: str) -> torch.Tensor:
        self.register_token(token)
        pattern = torch.zeros(self.config.n_units)
        pattern[self.output_assemblies[token]] = 1.0
        return pattern

    def simultaneous_state(
        self, tokens: Sequence[str], *, steps: int = 2, plastic: bool = False
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
            state = self.step(state, sensory=sensory, plastic=plastic)
        return state

    @staticmethod
    def assembly_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
        """Representational similarity; deliberately independent of 3-D distance."""
        return float(F.cosine_similarity(left[None], right[None]).item())

    def _cap_region_activity(self, activity: torch.Tensor) -> torch.Tensor:
        capped = activity.clone()
        for region_id in (INPUT, SUBSTRATE, OUTPUT):
            indices = torch.where(self.region == region_id)[0]
            density = self.config.max_region_density
            if region_id == OUTPUT and self.config.max_output_density is not None:
                density = self.config.max_output_density
            limit = max(1, round(len(indices) * density))
            values = capped[indices]
            if torch.count_nonzero(values) > limit:
                keep = torch.topk(values, limit).indices
                mask = torch.zeros_like(values, dtype=torch.bool)
                mask[keep] = True
                capped[indices[~mask]] = 0.0
        return capped

    def _adapt_firing_thresholds(self, activity: torch.Tensor) -> None:
        """Each neuron tracks its own rate and retunes its own threshold.

        Nothing here reads another neuron: no region rank, no region maximum,
        no region sum.

        Homeostasis keeps a neuron out of silence and out of runaway; it does
        not equalize rates.  Driving every rate to one set-point flattens the
        selectivity the category geometry is made of -- a unit that should
        answer only to animals gets its threshold dragged down until it answers
        to everything.  Cortical rates are broadly distributed, so the rule only
        acts outside a band and leaves the neuron alone inside it.

        This does not enforce a per-input density and is not a top-k in
        disguise.  It pins a neuron's rate over time; the fraction of neurons
        active on any one input is free, and measures far above the band here.
        """
        config = self.config
        fired = (activity > 0).float()
        self.firing_rate += config.firing_rate_trace * (fired - self.firing_rate)
        excess = (self.firing_rate - config.firing_rate_ceiling).clamp_min(0.0)
        deficit = (config.firing_rate_floor - self.firing_rate).clamp_min(0.0)
        delta = (
            config.threshold_adaptation_rate
            * self.intrinsic_plasticity
            * (excess - deficit)
        )
        self.firing_threshold += delta
        self.firing_threshold.clamp_(min=config.minimum_firing_threshold)
        self.threshold_change += delta.abs()

    @property
    def intrinsic_plasticity(self) -> torch.Tensor:
        """Per-neuron retuning availability from that neuron's own history.

        A threshold is a fast variable shared by every memory the neuron takes
        part in.  Left free it is a hole in the synaptic stability-plasticity
        solution: later learning retunes it and moves the R activity that older
        concepts were consolidated against.  It consolidates on the same
        warm->cold schedule the synapses use.
        """
        strength = self.config.intrinsic_stability_strength
        return 1.0 / (1.0 + strength * self.threshold_stability)

    def step(
        self,
        state: torch.Tensor,
        sensory: torch.Tensor | None = None,
        teacher: torch.Tensor | None = None,
        *,
        plastic: bool = False,
    ) -> torch.Tensor:
        drive = torch.zeros_like(state)
        weights = self.effective_weights
        if self.config.output_synaptic_scaling:
            output_edge = self.region[self.dst] == OUTPUT
            incoming = torch.zeros(self.config.n_units)
            incoming.scatter_add_(
                0, self.dst[output_edge], weights[output_edge].abs()
            )
            count = torch.zeros(self.config.n_units)
            count.scatter_add_(
                0,
                self.dst[output_edge],
                torch.ones(int(output_edge.sum().item())),
            )
            target_total = count * self.config.initial_weight
            gain = target_total / incoming.clamp_min(1e-8)
            weights = weights.clone()
            weights[output_edge] *= gain[self.dst[output_edge]]
        drive.scatter_add_(0, self.dst, state[self.src] * weights)
        potential = (1.0 - self.config.leak) * state + drive
        if self.config.homeostatic_threshold:
            activity = F.relu(potential - self.firing_threshold)
        else:
            activity = F.relu(potential - self.config.threshold)
            activity = self._cap_region_activity(activity)
        if sensory is not None:
            input_units = self.region == INPUT
            activity[input_units] = torch.maximum(activity[input_units], sensory[input_units])
        if teacher is not None:
            output_units = self.region == OUTPUT
            activity[output_units] = torch.maximum(activity[output_units], teacher[output_units])
        if self.config.homeostatic_threshold:
            # A neuron cannot fire harder than its own ceiling. This replaces
            # dividing the region by whichever neuron happened to fire most.
            activity = activity.clamp_max(1.0)
            if plastic:
                self._adapt_firing_thresholds(activity)
            return activity
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

    def predict_bound_output(
        self, bound_state: torch.Tensor, *, readout_steps: int = 1
    ) -> tuple[str, float, dict[str, float]]:
        """Decode an internally bound R state over the full O vocabulary."""
        if self.config.dendritic_output_enabled:
            state = self.dendritic_output_state(bound_state)
        else:
            state = bound_state.clone()
            for _ in range(readout_steps):
                state = self.step(state)
        self.state = state
        scores = self.output_scores(state, tuple(self.output_assemblies))
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
        output_error = self._output_error(target_pattern, readout, output_edge)
        source_plasticity = self._output_source_plasticity(
            target_pattern, output_edge
        )
        delta = (
            self.signs[output_edge]
            * free[self.src[output_edge]]
            * output_error
            * self.local_plasticity[output_edge]
            * source_plasticity
        )
        self.warm[output_edge] = (
            self.warm[output_edge] + learning_rate * delta
        ).clamp(min=-1.5, max=1.5)
        self._tag_target_output_edges(output_edge, free, target_pattern)

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
        output_error = self._output_error(target_pattern, readout, output_edge)
        source_plasticity = self._output_source_plasticity(
            target_pattern, output_edge
        )
        delta = (
            self.signs[output_edge]
            * free[self.src[output_edge]]
            * output_error
            * self.local_plasticity[output_edge]
            * source_plasticity
        )
        self.warm[output_edge] = (
            self.warm[output_edge] + learning_rate * delta
        ).clamp(min=-1.5, max=1.5)
        self._tag_target_output_edges(output_edge, free, target_pattern)
        self._learn_output_feedback(free, target_pattern, learning_rate)
        return float(delta.abs().mean().item())

    def learn_bound_output(
        self,
        bound_state: torch.Tensor,
        target: str,
        *,
        learning_rate: float = 0.03,
        structural_plasticity: bool = False,
    ) -> float:
        """Apply the local R->O rule to an internally formed bound state."""
        target_pattern = self.output_pattern(target)
        if self.config.dendritic_output_enabled:
            return self.learn_dendritic_output(bound_state, target_pattern)
        if structural_plasticity:
            self._rewire_output_inputs(bound_state, target_pattern)
        readout = self.step(bound_state)
        output_edge = (self.region[self.src] == SUBSTRATE) & (
            self.region[self.dst] == OUTPUT
        )
        output_error = self._output_error(target_pattern, readout, output_edge)
        source_plasticity = self._output_source_plasticity(
            target_pattern, output_edge
        )
        delta = (
            self.signs[output_edge]
            * bound_state[self.src[output_edge]]
            * output_error
            * self.local_plasticity[output_edge]
            * source_plasticity
        )
        self.warm[output_edge] = (
            self.warm[output_edge] + learning_rate * delta
        ).clamp(min=-1.5, max=1.5)
        self._tag_target_output_edges(output_edge, bound_state, target_pattern)
        self._learn_output_feedback(bound_state, target_pattern, learning_rate)
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
            threshold = self.config.output_commitment_threshold
            mature = self.stability[output_edges] >= threshold
            if (
                threshold > 0.0
                and bool(mature.any())
                and not bool(torch.isin(existing[mature], target_units).any())
            ):
                continue
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
            protection = (
                self.config.synaptic_stability_strength
                * self.stability[replacement_pool]
            )
            if self.config.structural_rewire_mode == "weakest":
                replacement = replacement_pool[
                    torch.topk(
                        magnitude + protection, number, largest=False
                    ).indices
                ]
            elif self.config.structural_rewire_mode == "local_stochastic":
                prune_probability = 1.0 / (magnitude + protection + 0.05)
                replacement = replacement_pool[
                    torch.multinomial(
                        prune_probability,
                        number,
                        replacement=False,
                        generator=self.generator,
                    )
                ]
            else:
                raise ValueError(
                    f"unknown structural_rewire_mode: "
                    f"{self.config.structural_rewire_mode}"
                )
            self.dst[replacement] = candidates[selected]
            self.cold[replacement] = 0.0
            self.warm[replacement] = self.config.initial_weight
            self.stability[replacement] = 0.0
            self.stability_tag[replacement] = 0.0

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
        self.warm.add_(
            learning_rate * delta * self.local_plasticity
        ).clamp_(min=-1.5, max=1.5)
        self.stability_tag[delta != 0] = 1.0
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
            ) * self.local_plasticity[entity_edge]
            self.warm.add_(learning_rate * delta).clamp_(min=-1.5, max=1.5)
            self.stability_tag[delta != 0] = 1.0
            if iteration % consolidate_every == 0:
                self.consolidate()

    def learn_sensory_concept(
        self,
        sensory_code: torch.Tensor,
        properties: Sequence[str],
        *,
        rounds: int = 40,
        learning_rate: float = 0.30,
        settle_steps: int = 2,
        consolidate_every: int = 10,
        structural_plasticity: bool = True,
    ) -> None:
        """Organize a learned sensory chunk without assigning a token name."""
        sensory = self.expanded_input_pattern(sensory_code)
        target = self.simultaneous_state(properties, steps=settle_steps)
        if structural_plasticity:
            self._rewire_concept_inputs(sensory, target)
        edge = (
            (self.region[self.src] == INPUT)
            & (self.region[self.dst] == SUBSTRATE)
            & (sensory[self.src] > 0)
        )
        for iteration in range(rounds):
            free = self.sensory_assembly_state(
                sensory_code, steps=settle_steps, plastic=True
            )
            target = self.simultaneous_state(
                properties, steps=settle_steps, plastic=True
            )
            delta = torch.zeros_like(self.warm)
            delta[edge] = (
                target[self.dst[edge]] - free[self.dst[edge]]
            ) * self.local_plasticity[edge]
            self.warm.add_(learning_rate * delta).clamp_(min=-1.5, max=1.5)
            self.stability_tag[delta != 0] = 1.0
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
            replaceable = torch.ones(len(source_edges), dtype=torch.bool)
            threshold = self.config.concept_rewire_stability_threshold
            if threshold > 0.0:
                replaceable = self.stability[source_edges] < threshold
            number = min(
                count,
                len(candidates),
                int(replaceable.sum().item()),
            )
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

            if self.config.structural_rewire_mode == "weakest":
                replacement_score = (
                    target[existing]
                    + self.config.synaptic_stability_strength
                    * self.stability[source_edges]
                )
                replacement_score[~replaceable] = float("inf")
                replacement = torch.topk(
                    replacement_score,
                    number,
                    largest=False,
                ).indices
            elif self.config.structural_rewire_mode == "local_stochastic":
                prune_probability = 1.0 / (
                    target[existing]
                    + self.config.synaptic_stability_strength
                    * self.stability[source_edges]
                    + 0.05
                )
                prune_probability[~replaceable] = 0.0
                replacement = torch.multinomial(
                    prune_probability,
                    number,
                    replacement=False,
                    generator=self.generator,
                )
            else:
                raise ValueError(
                    f"unknown structural_rewire_mode: "
                    f"{self.config.structural_rewire_mode}"
                )
            edge_indices = source_edges[replacement]
            self.dst[edge_indices] = new_destinations
            self.cold[edge_indices] = self.config.initial_weight
            self.warm[edge_indices] = 0.0
            self.stability[edge_indices] = 0.0
            self.stability_tag[edge_indices] = 0.0

    def consolidate(self, cycles: int = 1) -> None:
        """Transfer fast synaptic change into the slow variable on each edge."""
        if cycles < 0:
            raise ValueError("cycles must be non-negative")
        rate = self.config.consolidation_rate
        for _ in range(cycles):
            transfer = rate * self.warm
            self.cold.add_(transfer).clamp_(min=0.0, max=2.0)
            self.stability.add_(
                self.config.synaptic_stability_rate
                * transfer.abs()
                * self.stability_tag
            ).clamp_(max=2.0)
            self.warm.mul_(self.config.warm_decay)
            self.stability_tag.mul_(self.config.warm_decay)
            # A neuron locks its own excitability in proportion to how much it
            # actually retuned, the way an edge's stability follows its own
            # transfer. Locking every neuron on every cycle regardless of what
            # it did freezes the thresholds at their initial value after the
            # first episode, which silently turns homeostasis off.
            self.threshold_stability.add_(
                self.config.intrinsic_stability_rate
                * rate
                * self.threshold_change.abs()
            ).clamp_(max=2.0)
            self.threshold_change.mul_(self.config.warm_decay)
            for branches in self.output_dendritic_branches.values():
                for branch in branches:
                    branch.cold = min(
                        2.0, branch.cold + rate * branch.warm
                    )
                    branch.warm *= self.config.warm_decay

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
