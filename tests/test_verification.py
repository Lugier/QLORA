import unittest
from unittest.mock import patch

from verification import assess_test_quality, is_test_quality_sufficient, run_test_verifier


class VerificationTests(unittest.TestCase):
    def test_quality_metrics_non_empty(self):
        tests = "assert foo(1) == 2\nassert foo(2) == 3"
        metrics = assess_test_quality(tests)
        self.assertGreaterEqual(metrics["assert_count"], 2)
        self.assertGreater(metrics["quality_score"], 0.0)
        self.assertTrue(is_test_quality_sufficient(tests, min_asserts=2, min_nonempty_lines=2, min_quality_score=1.0))

    def test_fractional_execution_reward_signal(self):
        code = "def add(a, b):\n    return a + b\n"
        tests = "assert add(1, 2) == 3\nassert add(2, 2) == 5"
        result = run_test_verifier(code=code, tests=tests, rounds=1, timeout=1.5, require_all_pass=True)
        self.assertFalse(result["all_passed"])
        self.assertGreater(result["pass_fraction"], 0.0)
        self.assertLess(result["pass_fraction"], 1.0)

    def test_all_passed_path(self):
        code = "def mul(a, b):\n    return a * b\n"
        tests = "assert mul(2, 3) == 6\nassert mul(0, 3) == 0"
        result = run_test_verifier(code=code, tests=tests, rounds=1, timeout=1.5, require_all_pass=True)
        self.assertTrue(result["all_passed"])
        self.assertEqual(result["score"], 2.0)
        self.assertGreaterEqual(result["pass_fraction"], 1.0)

    def test_security_violation_recorded(self):
        code = "import os\n\ndef f():\n    return 1\n"
        tests = "assert f() == 1"
        result = run_test_verifier(code=code, tests=tests, rounds=1, timeout=1.5, require_all_pass=True)
        self.assertFalse(result["all_passed"])
        self.assertIn("security", result["error_counts"])

    def test_fractional_fallback_from_round_scores(self):
        code = "def add(a, b):\n    return a + b\n"
        tests = "assert add(1, 2) == 3\nassert add(2, 2) == 5"
        with patch("pipeline.core.verification._instrument_tests_for_fractional_counts", return_value=""):
            result = run_test_verifier(code=code, tests=tests, rounds=1, timeout=1.5, require_all_pass=True)
        self.assertEqual(result["fractional_mode"], "fallback_round_scores")
        self.assertGreater(result["pass_fraction"], 0.0)

    def test_timeout_retry_recovers_score(self):
        code = "def mul(a, b):\n    return a * b\n"
        tests = "assert mul(2, 3) == 6"
        timeout_result = {
            "score": -0.5,
            "exec_time": 1.0,
            "stdout": "",
            "stderr": "Execution timed out.",
            "error_type": "timeout",
        }
        success_result = {
            "score": 2.0,
            "exec_time": 0.2,
            "stdout": "",
            "stderr": "",
            "error_type": "",
        }

        # First run times out; retry should recover.
        with patch("pipeline.core.verification.run_code_in_sandbox_detailed", side_effect=[timeout_result, success_result, success_result]):
            result = run_test_verifier(
                code=code,
                tests=tests,
                rounds=1,
                timeout=1.0,
                require_all_pass=True,
                retry_on_timeout=True,
                timeout_retry_factor=1.5,
            )
        self.assertTrue(result["all_passed"])
        self.assertEqual(result["score"], 2.0)


if __name__ == "__main__":
    unittest.main()
