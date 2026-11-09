import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/classify_matched_grasp_events.py"
spec = importlib.util.spec_from_file_location("classify_matched_grasp_events", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EventClassificationTests(unittest.TestCase):
    def test_precedence_and_boundaries(self):
        classify = module.classify_episode
        self.assertEqual(classify(True, 10, 0.1, 100), "verified_grasp")
        self.assertEqual(classify(False, None, 0.1, None), "no_close_before_budget")
        self.assertEqual(classify(False, 10, 0.02, 80), "object_displaced_before_close")
        self.assertEqual(classify(False, 10, 0.019, 40), "off_target_at_close")
        self.assertEqual(classify(False, 10, 0.0, 39.9), "failed_close_near_target")


if __name__ == "__main__":
    unittest.main()
