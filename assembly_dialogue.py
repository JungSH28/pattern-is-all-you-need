"""Two assembly-based dialogue tracks with one fact/query protocol.

The functional track deliberately permits global associative updates. The
bio-local track wraps SpatialConnectome and uses local R->O value updates, while
reporting the remaining non-local scaffolds instead of hiding them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import torch

from spatial_connectome import SpatialConnectome


@dataclass(frozen=True)
class LocalityAudit:
    sparse_token_assemblies: bool
    local_state_propagation: bool
    local_synaptic_value_update: bool
    local_query_gate_application: bool
    source_local_stochastic_rewiring: bool
    no_autograd_or_weight_transport: bool
    global_teacher_target: bool
    global_activity_competition: bool
    global_structural_search: bool
    global_seed_and_gate_balancing: bool

    def as_dict(self) -> dict[str, bool]:
        return asdict(self)


class FunctionalAssemblyDialogue:
    """Global associative ceiling that keeps tokens as sparse assemblies."""

    def __init__(self, *, n_units: int = 512, assembly_size: int = 16, seed: int = 0):
        self.n_units = n_units
        self.assembly_size = assembly_size
        self.generator = torch.Generator().manual_seed(seed)
        self.codes: dict[str, torch.Tensor] = {}
        self.seed_usage = torch.zeros(n_units, dtype=torch.long)
        self.query_gates: dict[str, torch.Tensor] = {}
        self.semantic_memory = torch.zeros(n_units, n_units)
        self.output_memory = torch.zeros(n_units, n_units)
        self.state = torch.zeros(n_units)

    def register_token(self, token: str) -> None:
        if token in self.codes:
            return
        tie_break = torch.rand(self.n_units, generator=self.generator)
        units = torch.argsort(
            self.seed_usage.float() + 1e-3 * tie_break
        )[: self.assembly_size]
        code = torch.zeros(self.n_units)
        code[units] = 1.0
        self.codes[token] = code
        self.seed_usage[units] += 1

    def register_vocabulary(self, tokens: Iterable[str]) -> None:
        for token in tokens:
            self.register_token(token)

    def register_queries(self, queries: Sequence[str]) -> None:
        if not queries:
            raise ValueError("queries must not be empty")
        for query in queries:
            self.register_token(query)
        permutation = torch.randperm(self.n_units, generator=self.generator)
        for query, units in zip(queries, torch.tensor_split(permutation, len(queries))):
            gate = torch.zeros(self.n_units)
            gate[units] = 1.0
            self.query_gates[query] = gate

    def observe_fact(self, entity: str, properties: Sequence[str]) -> None:
        self.register_vocabulary((entity, *properties))
        entity_code = self.codes[entity] / self.assembly_size
        property_code = torch.stack([self.codes[token] for token in properties]).sum(0)
        self.semantic_memory.add_(torch.outer(entity_code, property_code))

    def query_state(self, query: str, entity: str, *, gated: bool = True) -> torch.Tensor:
        if query not in self.query_gates:
            raise KeyError(f"unregistered query: {query}")
        self.register_token(entity)
        state = self.codes[entity] @ self.semantic_memory
        if gated:
            state = state * self.query_gates[query]
        self.state = state
        return state

    def teach_answer(self, query: str, entity: str, target: str) -> None:
        self.register_token(target)
        state = self.query_state(query, entity)
        scale = state.square().sum().clamp_min(1e-6)
        self.output_memory.add_(
            torch.outer(state / scale, self.codes[target])
        )

    def answer(
        self, query: str, entity: str, *, gated: bool = True
    ) -> tuple[str, float, dict[str, float]]:
        state = self.query_state(query, entity, gated=gated)
        output = state @ self.output_memory
        scores = {token: float(output @ code) for token, code in self.codes.items()}
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
        return ordered[0][0], ordered[0][1] - runner_up, scores

    def clear_hot_state(self) -> None:
        self.state.zero_()


class BioLocalAssemblyDialogue:
    """Dialogue facade for the single sparse connectome."""

    def __init__(self, connectome: SpatialConnectome):
        self.connectome = connectome

    def register_vocabulary(self, tokens: Iterable[str]) -> None:
        self.connectome.register_vocabulary(tokens)

    def register_queries(self, queries: Sequence[str]) -> None:
        for query in queries:
            self.connectome.register_query(query)

    def observe_fact(self, entity: str, properties: Sequence[str]) -> None:
        self.connectome.learn_concept(entity, properties)

    def teach_answer(
        self,
        query: str,
        entity: str,
        target: str,
        *,
        structural_plasticity: bool = False,
    ) -> None:
        self.connectome.learn_query_association(
            query,
            entity,
            target,
            structural_plasticity=structural_plasticity,
        )

    def answer(
        self, query: str, entity: str, *, gated: bool = True
    ) -> tuple[str, float, dict[str, float]]:
        return self.connectome.predict_query_output(
            query, entity, gated=gated
        )

    def consolidate_and_clear(self, *, cycles: int = 100) -> None:
        self.connectome.consolidate(cycles=cycles)
        self.connectome.warm.zero_()
        self.connectome.state.zero_()

    @staticmethod
    def locality_audit() -> LocalityAudit:
        return LocalityAudit(
            sparse_token_assemblies=True,
            local_state_propagation=True,
            local_synaptic_value_update=True,
            local_query_gate_application=True,
            source_local_stochastic_rewiring=True,
            no_autograd_or_weight_transport=True,
            global_teacher_target=True,
            global_activity_competition=True,
            global_structural_search=False,
            global_seed_and_gate_balancing=True,
        )
