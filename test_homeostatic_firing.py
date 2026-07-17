"""Structure tests for homeostatic threshold firing.

These pin the claim that replaces the global activity scaffold: a neuron fires
on its own threshold, retunes it from its own rate, and nothing in the firing
path ranks, maximizes or sums a region.
"""

from __future__ import annotations

import inspect
import unittest

import torch

from spatial_connectome import (
    OUTPUT,
    SUBSTRATE,
    ConnectomeConfig,
    SpatialConnectome,
)
from syllable_chunk_dialogue import ConnectomeSyllableDialogue


def _homeostatic() -> SpatialConnectome:
    return SpatialConnectome(
        ConnectomeConfig(homeostatic_threshold=True, seed=0)
    )


class FiringPathIsLocal(unittest.TestCase):
    def test_adaptation_reads_only_the_neuron_itself(self):
        source = inspect.getsource(SpatialConnectome._adapt_firing_thresholds)
        for global_op in ("topk", "argsort", "sort(", ".max()", ".sum()", ".mean()"):
            self.assertNotIn(global_op, source, msg=global_op)

    def test_homeostatic_step_never_ranks_or_caps_a_region(self):
        model = _homeostatic()
        calls: list[str] = []
        model._cap_region_activity = lambda activity: calls.append("capped") or activity
        state = torch.zeros(model.config.n_units)
        state[model.region == SUBSTRATE] = 1.0
        model.step(state, plastic=True)
        self.assertEqual(calls, [])

    def test_audit_no_longer_claims_global_activity_competition(self):
        audit = ConnectomeSyllableDialogue.locality_audit().as_dict()
        self.assertFalse(audit["global_sparse_activity_competition"])
        self.assertTrue(audit["homeostatic_threshold_firing"])
        self.assertTrue(audit["intrinsic_excitability_consolidation"])


class ThresholdsBehave(unittest.TestCase):
    def test_runaway_and_silent_neurons_retune_in_opposite_directions(self):
        model = _homeostatic()
        before = model.firing_threshold.clone()
        active = torch.zeros(model.config.n_units)
        active[0] = 1.0
        # The rate trace starts inside the band, and it is deliberately slow, so
        # leaving the band at all takes hundreds of steps.
        for _ in range(2000):
            model._adapt_firing_thresholds(active)
        self.assertGreater(float(model.firing_threshold[0]), float(before[0]))
        self.assertLess(float(model.firing_threshold[1]), float(before[1]))

    def test_active_count_varies_with_the_input(self):
        """No quota: k is whatever crosses threshold."""
        model = _homeostatic()
        counts = set()
        for scale in (0.2, 0.5, 1.0, 2.0):
            state = torch.zeros(model.config.n_units)
            state[model.region == SUBSTRATE] = scale
            activity = model.step(state)
            counts.add(int((activity[model.region == OUTPUT] > 0).sum()))
        self.assertGreater(len(counts), 1)

    def test_thresholds_actually_move_during_training(self):
        """The mechanism must not be inert.

        Locking every neuron on every consolidation cycle regardless of what it
        did froze the thresholds at their initial value after the first fact
        episode. Everything still passed, because a frozen per-neuron threshold
        is just the old fixed threshold, so nothing failed and the homeostasis
        was silently doing nothing. Pin the movement itself.
        """
        from category_generalization_probe import CONCEPTS, LABELED
        from syllable_chunk_probe import fact_episodes, stage_streams
        from syllable_sentence_dialogue import ConnectomeSentenceDialogue
        from syllable_sentence_probe import sentence_vocabulary_texts

        model = ConnectomeSentenceDialogue(seed=0)
        model.register_syllable_vocabulary(sentence_vocabulary_texts())
        model.register_syllable_output(sentence_vocabulary_texts())
        model.observe_streams(stage_streams(CONCEPTS, LABELED))
        model.finalize_chunks()
        connectome = model.connectome
        start = connectome.firing_threshold.clone()
        for episode in fact_episodes(CONCEPTS):
            model.observe_fact_episode(episode, rounds=30)

        substrate = connectome.region == SUBSTRATE
        moved = (connectome.firing_threshold - start)[substrate]
        self.assertGreater(float(moved.abs().mean()), 0.02)
        # And they must differentiate, not drift as one block.
        self.assertGreater(float(connectome.firing_threshold[substrate].std()), 0.001)

    def test_consolidation_locks_intrinsic_excitability(self):
        model = SpatialConnectome(
            ConnectomeConfig(
                homeostatic_threshold=True,
                intrinsic_stability_strength=100.0,
                seed=0,
            )
        )
        self.assertTrue(bool((model.intrinsic_plasticity == 1.0).all()))
        # Locking follows the neuron's own retuning, the way an edge's
        # stability follows its own transfer. A neuron that never retuned stays
        # plastic no matter how many cycles pass.
        model.consolidate(cycles=100)
        self.assertTrue(bool((model.intrinsic_plasticity == 1.0).all()))
        model.threshold_change += 1.0
        model.consolidate(cycles=100)
        self.assertTrue(bool((model.intrinsic_plasticity < 0.02).all()))

        # A consolidated neuron retunes far more slowly, which is what protects
        # an older concept's R activity from later learning. Like the synaptic
        # rule this damps rather than freezes; freezing it outright is what
        # turned the whole mechanism into a fixed threshold.
        active = torch.ones(model.config.n_units)

        def drift(connectome: SpatialConnectome) -> float:
            start = connectome.firing_threshold.clone()
            for _ in range(400):
                connectome._adapt_firing_thresholds(active)
            return float((connectome.firing_threshold - start).abs().mean())

        free = _homeostatic()
        self.assertLess(drift(model), 0.02 * drift(free))


if __name__ == "__main__":
    unittest.main()
