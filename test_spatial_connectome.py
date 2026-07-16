import unittest

import torch

from spatial_connectome import (
    INPUT,
    OUTPUT,
    SUBSTRATE,
    ConnectomeConfig,
    SpatialConnectome,
)


class SpatialConnectomeTests(unittest.TestCase):
    def setUp(self):
        self.model = SpatialConnectome(
            ConnectomeConfig(
                n_input=24,
                n_substrate=48,
                n_output=24,
                out_degree=12,
                steps_per_token=4,
                seed=7,
            )
        )
        self.model.register_vocabulary(["a", "b", "c", "animal", "cat", "dog"])

    def test_regions_are_one_contiguous_connectome(self):
        c = self.model.config
        self.assertTrue(torch.all(self.model.region[: c.n_input] == INPUT))
        self.assertTrue(
            torch.all(
                self.model.region[c.n_input : c.n_input + c.n_substrate] == SUBSTRATE
            )
        )
        self.assertTrue(torch.all(self.model.region[-c.n_output :] == OUTPUT))
        self.assertEqual(self.model.sparse_matrix().shape, (c.n_units, c.n_units))
        self.assertEqual(len(self.model.src), c.n_units * c.out_degree)

    def test_each_unit_vector_is_a_row_of_the_same_graph(self):
        matrix = self.model.sparse_matrix().to_dense()
        for unit in (0, self.model.config.n_input, self.model.config.n_units - 1):
            self.assertTrue(torch.equal(matrix[unit], self.model.outgoing_vector(unit)))

    def test_initial_token_seeds_avoid_accidental_overlap(self):
        input_units = torch.cat(list(self.model.input_assemblies.values()))
        output_units = torch.cat(list(self.model.output_assemblies.values()))
        self.assertEqual(len(torch.unique(input_units)), len(input_units))
        self.assertEqual(len(torch.unique(output_units)), len(output_units))

    def test_larger_vocabulary_never_reuses_exact_seed_assembly(self):
        self.model.register_vocabulary(f"token_{index}" for index in range(80))
        input_codes = {
            tuple(sorted(units.tolist()))
            for units in self.model.input_assemblies.values()
        }
        output_codes = {
            tuple(sorted(units.tolist()))
            for units in self.model.output_assemblies.values()
        }
        self.assertEqual(len(input_codes), len(self.model.input_assemblies))
        self.assertEqual(len(output_codes), len(self.model.output_assemblies))

    def test_distance_changes_topology(self):
        self.assertLess(
            self.model.mean_edge_distance(), self.model.mean_allowed_distance()
        )

    def test_random_topology_removes_distance_bias(self):
        random_model = SpatialConnectome(
            ConnectomeConfig(
                n_input=24,
                n_substrate=48,
                n_output=24,
                out_degree=12,
                topology="random",
                seed=7,
            )
        )
        distance_gap = abs(
            random_model.mean_edge_distance() - random_model.mean_allowed_distance()
        )
        biased_gap = abs(
            self.model.mean_edge_distance() - self.model.mean_allowed_distance()
        )
        self.assertLess(distance_gap, biased_gap)

    def test_dale_sign_is_fixed_per_source(self):
        weight = self.model.effective_weights
        for unit in range(self.model.config.n_units):
            outgoing = weight[self.model.src == unit]
            self.assertTrue(torch.all(outgoing >= 0) or torch.all(outgoing <= 0))

    def test_sequence_order_changes_hot_state(self):
        ab = self.model.run_sequence(["a", "b"])
        ba = self.model.run_sequence(["b", "a"])
        self.assertIsInstance(ab, torch.Tensor)
        self.assertIsInstance(ba, torch.Tensor)
        self.assertGreater(torch.linalg.vector_norm(ab - ba).item(), 1e-4)

    def test_cosine_measures_assembly_not_physical_distance(self):
        left = torch.zeros(self.model.config.n_units)
        right = torch.zeros_like(left)
        left[[0, 1, 2, 3]] = 1
        right[[2, 3, 4, 5]] = 1
        self.assertAlmostEqual(self.model.assembly_cosine(left, right), 0.5, places=6)

    def test_position_plasticity_moves_then_decays(self):
        samples = []
        for sequence in (["cat", "animal"], ["dog", "animal"]):
            _, trace = self.model.run_sequence(sequence, keep_trace=True)
            samples.extend(trace)
        before = self.model.positions.clone()
        plasticity = self.model.position_plasticity
        self.model.develop_positions(torch.stack(samples), epochs=1)
        self.assertGreater(torch.linalg.vector_norm(self.model.positions - before).item(), 0)
        self.assertLess(self.model.position_plasticity, plasticity)

    def test_warm_change_can_consolidate_into_cold_memory(self):
        self.model.learn_association(["a", "b"], "c")
        self.assertGreater(self.model.warm.abs().sum().item(), 0)
        cold_before = self.model.cold.clone()
        warm_before = self.model.warm.clone()
        self.model.consolidate()
        self.assertGreater(torch.linalg.vector_norm(self.model.cold - cold_before).item(), 0)
        self.assertLess(
            self.model.warm.abs().sum().item(), warm_before.abs().sum().item()
        )

    def test_property_learning_changes_entity_assembly(self):
        self.model.register_vocabulary(["fur", "fourlegs"])
        before = self.model.simultaneous_state(("cat",))
        destinations_before = self.model.dst.clone()
        self.model.learn_concept("cat", ("fur", "fourlegs"), rounds=20)
        after = self.model.simultaneous_state(("cat",))
        self.assertGreater(torch.linalg.vector_norm(after - before).item(), 1e-4)
        self.assertGreater(
            torch.count_nonzero(self.model.dst != destinations_before).item(), 0
        )
        self.assertEqual(len(self.model.src), len(destinations_before))

    def test_local_learning_increases_target_output(self):
        before_state = self.model.run_sequence(["a", "b"])
        before = self.model.output_scores(before_state, ["c"])["c"]
        for iteration in range(120):
            self.model.learn_association(["a", "b"], "c", learning_rate=0.10)
            if iteration % 10 == 0:
                self.model.consolidate()
        after_state = self.model.run_sequence(["a", "b"])
        after = self.model.output_scores(after_state, ["c"])["c"]
        self.assertGreater(after, before + 0.05)

    def test_category_output_can_consolidate_into_cold_memory(self):
        edge_count = len(self.model.src)
        for iteration in range(40):
            self.model.learn_output_association(
                ("cat",),
                "animal",
                structural_plasticity=iteration == 0,
            )
        warm_prediction = self.model.predict_output(
            ("cat",), ("animal", "dog")
        )[0]
        self.assertEqual(warm_prediction, "animal")
        self.assertEqual(len(self.model.src), edge_count)

        self.model.consolidate(cycles=100)
        self.model.warm.zero_()
        self.model.state.zero_()
        cold_prediction = self.model.predict_output(
            ("cat",), ("animal", "dog")
        )[0]
        self.assertEqual(cold_prediction, "animal")

    def test_query_gates_separate_same_entity_contexts(self):
        self.model.register_query("what_is")
        self.model.register_query("what_feature")
        cat = self.model.query_bound_state("what_is", "cat")
        feature = self.model.query_bound_state("what_feature", "cat")
        substrate = self.model.region == SUBSTRATE
        self.assertGreater(torch.linalg.vector_norm(cat - feature).item(), 1e-4)
        self.assertEqual(
            torch.count_nonzero(
                self.model.query_gates["what_is"][substrate]
                * self.model.query_gates["what_feature"][substrate]
            ).item(),
            0,
        )

    def test_query_answer_uses_entire_output_vocabulary(self):
        self.model.learn_concept("cat", ("fur", "fourlegs"), rounds=20)
        for iteration in range(40):
            self.model.learn_query_association(
                "what_is",
                "cat",
                "animal",
                structural_plasticity=iteration == 0,
            )
        predicted, _, scores = self.model.predict_query_output("what_is", "cat")
        self.assertEqual(set(scores), set(self.model.output_assemblies))
        self.assertEqual(predicted, "animal")

    def test_output_can_use_a_different_local_competition_density(self):
        model = SpatialConnectome(
            ConnectomeConfig(
                n_input=24,
                n_substrate=48,
                n_output=24,
                out_degree=12,
                max_region_density=0.05,
                max_output_density=0.25,
                seed=9,
            )
        )
        activity = torch.ones(model.config.n_units)
        capped = model._cap_region_activity(activity)
        self.assertEqual(
            torch.count_nonzero(capped[model.region == SUBSTRATE]).item(), 2
        )
        self.assertEqual(
            torch.count_nonzero(capped[model.region == OUTPUT]).item(), 6
        )


if __name__ == "__main__":
    unittest.main()
