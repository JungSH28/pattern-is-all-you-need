"""Answering from two facts at once, not from one retrieval.

Every earlier goal answers by retrieval: one entity assembly is evoked, gated
by the query, and the stored answer comes back.  A question about what two
entities have in common has no stored answer for any pair -- it exists only in
what both entity assemblies drive together, and for two entities from different
categories the honest answer is that there is nothing.

The mechanism under test is that co-activating two concept chunks lets the R
units both of them drive stand out, so the shared property is what the readout
sees.  Nothing here searches or intersects the two fact sets; the substrate is
asked to do it.
"""

from __future__ import annotations

from typing import Sequence

import torch

from syllable_chunk_dialogue import _occurrences, syllable_sequence
from syllable_sentence_dialogue import ConnectomeSentenceDialogue


class ConnectomeCompositionDialogue(ConnectomeSentenceDialogue):
    """Co-activate every concept the question names.

    ``use_replay`` is off: a common-property answer names no entity, so the
    repetition pathway has nothing to contribute.  Selecting it per query type
    is an audited scaffold, not a learned routing.
    """

    def __init__(self, *, seed: int = 0, **kwargs):
        kwargs.setdefault("use_replay", False)
        super().__init__(seed=seed, **kwargs)

    def entity_patterns(self, text: str) -> tuple[tuple[str, ...], ...]:
        """Every learned concept chunk the question contains."""
        sequence = syllable_sequence(text)
        present = [
            pattern
            for pattern in self.concept_patterns
            if _occurrences(sequence, pattern)
        ]
        if not present:
            raise RuntimeError("question contains no learned concept chunk")
        # A shorter chunk nested inside a longer one is the same object named
        # twice, not a second object.
        return tuple(
            pattern
            for pattern in present
            if not any(
                other != pattern and _occurrences(other, pattern)
                for other in present
            )
        )

    def _entity_pattern(self, text: str) -> tuple[str, ...]:
        return max(self.entity_patterns(text), key=len)

    def _control_pattern(
        self, text: str, entity: tuple[str, ...]
    ) -> torch.Tensor:
        """Query control must exclude every named entity, not just one.

        The single-entity version masks the spans of the entity it is given, so
        with two entities a chunk carrying the other one's syllables would be
        read as query wording.
        """
        sequence = syllable_sequence(text)
        spans = [
            (start, start + len(pattern))
            for pattern in self.entity_patterns(text)
            for start in _occurrences(sequence, pattern)
        ]
        controls = []
        for pattern, code, _ in self.chunker.matched_codes(text):
            if not 2 <= len(pattern) <= 3 or pattern in self.concept_patterns:
                continue
            if any(
                start < entity_end and entity_start < start + len(pattern)
                for start in _occurrences(sequence, pattern)
                for entity_start, entity_end in spans
            ):
                continue
            controls.append(code)
        if not controls:
            return self.chunker.feature(text, temporal=False)
        from syllable_chunk_dialogue import _sparsify_and_normalize

        return _sparsify_and_normalize(
            torch.stack(controls).sum(0), max_active=None
        )

    def bound_state(self, text: str, *, controlled: bool = True) -> torch.Tensor:
        """Evoke R from all named concepts at once, then gate by the query.

        Units that only one concept drives stay near their own threshold, while
        units both drive are pushed further past it. The shared meaning is not
        computed and inserted; it is what survives.
        """
        patterns = self.entity_patterns(text)
        code = self.concept_patterns[patterns[0]]
        for pattern in patterns[1:]:
            code = torch.maximum(code, self.concept_patterns[pattern])
        state = self.connectome.sensory_assembly_state(code, steps=2)
        return self._gate(text, patterns, state, controlled=controlled)

    def _gate(
        self,
        text: str,
        patterns: Sequence[tuple[str, ...]],
        state: torch.Tensor,
        *,
        controlled: bool,
    ) -> torch.Tensor:
        from spatial_connectome import SUBSTRATE

        bound = torch.zeros_like(state)
        substrate = self.connectome.region == SUBSTRATE
        if not controlled:
            bound[substrate] = state[substrate]
            return bound
        gate = self._learned_control_gate(text, max(patterns, key=len))
        bound[substrate] = state[substrate] * gate[substrate]
        return bound
