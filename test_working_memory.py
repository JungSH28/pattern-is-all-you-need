"""Structure tests for recurrent working memory (goal #36).

Pin that maintenance is local, that a held state decodes to its own item, that
several items are held without merging, and that recurrence is what does it.
"""

from __future__ import annotations

import inspect
import unittest

import torch

from spatial_connectome import SUBSTRATE, SpatialConnectome
from working_memory_probe import decode, recall, result, train


class MaintenanceIsLocal(unittest.TestCase):
    def test_hold_reads_only_pre_and_post_activity(self):
        source = inspect.getsource(SpatialConnectome.hold_assembly)
        for global_op in ("topk", "argsort", "sort(", ".max()", ".sum()", ".mean()"):
            self.assertNotIn(global_op, source, msg=global_op)

    def test_recurrent_edge_is_substrate_to_substrate_only(self):
        model = train(0)
        connectome = model.connectome
        edge = connectome.recurrent_edge
        self.assertTrue(bool((connectome.region[connectome.src[edge]] == SUBSTRATE).all()))
        self.assertTrue(bool((connectome.region[connectome.dst[edge]] == SUBSTRATE).all()))

    def test_hold_only_strengthens_edges_between_co_active_units(self):
        model = train(0, recurrent=False)
        connectome = model.connectome
        edge = connectome.recurrent_edge
        state = torch.zeros(connectome.config.n_units)
        substrate = torch.where(connectome.region == SUBSTRATE)[0]
        active = substrate[:8]
        state[active] = 1.0
        before = connectome.cold[edge].clone()
        connectome.hold_assembly(state, rate=1.0)
        changed = connectome.cold[edge] != before
        both_active = (state[connectome.src[edge]] > 0) & (state[connectome.dst[edge]] > 0)
        self.assertTrue(bool((changed == both_active).all()))


class Maintenance(unittest.TestCase):
    def test_recurrence_holds_and_control_does_not(self):
        blank = 5
        held = sum(int(result(seed, blank=blank)["correct"]) for seed in range(3))
        control = sum(
            int(result(seed, blank=blank, recurrent=False)["correct"])
            for seed in range(3)
        )
        # 18 items over 3 seeds; chance is 3. Recurrence must clear most of them
        # and the control must stay near chance.
        self.assertGreaterEqual(held, 14)
        self.assertLessEqual(control, 7)

    def test_items_are_held_apart_not_merged(self):
        # A single-attractor collapse would push every held state together.
        rows = [result(seed, blank=5) for seed in range(3)]
        for row in rows:
            self.assertLess(float(row["max_cross_cosine"]), 0.9)

    def test_a_held_state_decodes_to_its_own_item(self):
        model = train(0)
        for item in model.patterns:
            self.assertEqual(decode(model, recall(model, item, blank=3)), item)


if __name__ == "__main__":
    unittest.main()
