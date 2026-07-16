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
            probability = torch.exp(-distance / c.distance_scale) + c.long_range_floor
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
        inp = torch.randperm(c.n_input, generator=self.generator)[:k_input]
        out = torch.randperm(c.n_output, generator=self.generator)[:k_output]
        self.input_assemblies[token] = inp
        self.output_assemblies[token] = out + c.n_input + c.n_substrate

    def register_vocabulary(self, tokens: Iterable[str]) -> None:
        for token in tokens:
            self.register_token(token)

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
        self.warm.add_(learning_rate * delta).clamp_(min=0.0, max=1.5)
        return float(delta.abs().mean().item())

    def consolidate(self) -> None:
        """Transfer fast synaptic change into the slow variable on each edge."""
        rate = self.config.consolidation_rate
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
