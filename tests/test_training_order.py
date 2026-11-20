import unittest
import numpy as np
from jaka_vla.training_order import sample_plan


class SampleOrderTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_across_epochs(self):
        ids = [2, 5, 7, 11, 17]
        full = sample_plan(ids, 19, 42)
        resumed = np.concatenate([sample_plan(ids, 7, 42), sample_plan(ids, 12, 42, 7)])
        np.testing.assert_array_equal(full, resumed)
        for start in (0, 5, 10):
            self.assertEqual(set(full[start:start+5]), set(ids))

    def test_no_overlap_before_full_pass(self):
        first = sample_plan(range(100), 30, 47)
        second = sample_plan(range(100), 70, 47, 30)
        self.assertFalse(set(first) & set(second))
        self.assertEqual(len(set(first) | set(second)), 100)

    def test_reject_invalid_inputs(self):
        for ids, count, offset in [([], 1, 0), ([1, 1], 2, 0), ([1], 0, 0), ([1], 2, -1)]:
            with self.assertRaises(ValueError):
                sample_plan(ids, count, 42, offset)
