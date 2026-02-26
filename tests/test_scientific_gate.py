import unittest

from scripts.scientific_gate import _align_diffs, _bootstrap_weighted_diff_ci, _sign_flip_pvalue


class ScientificGateTests(unittest.TestCase):
    def test_align_diffs_pass_metric(self):
        base = [
            {"benchmark": "livecodebench", "mode": "code", "id": "1", "pass_at_1": False},
            {"benchmark": "livecodebench", "mode": "code", "id": "2", "pass_at_1": True},
        ]
        post = [
            {"benchmark": "livecodebench", "mode": "code", "id": "1", "pass_at_1": True},
            {"benchmark": "livecodebench", "mode": "code", "id": "2", "pass_at_1": True},
        ]
        diffs = _align_diffs(base, post, metric="pass", benchmark="livecodebench")
        self.assertEqual(diffs, [1.0, 0.0])

    def test_weighted_ci_and_pvalue(self):
        diffs = {
            "livecodebench": [1.0, 1.0, 0.0, 1.0],
            "bigcodebench_instruct": [1.0, 0.0, 1.0, 0.0],
            "swebench_verified_subset": [1.0, 1.0, 1.0, 0.0],
            "agentic_success": [1.0, 0.0, 1.0, 0.0],
            "format_quality": [0.2, 0.1, 0.3, 0.2],
        }
        mean, lo, hi = _bootstrap_weighted_diff_ci(diffs, bootstrap_samples=200, seed=3407)
        self.assertGreaterEqual(mean, lo)
        self.assertLessEqual(mean, hi)
        p = _sign_flip_pvalue(diffs, observed_diff=mean, samples=200, seed=3407)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)


if __name__ == "__main__":
    unittest.main()
