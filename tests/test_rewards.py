import unittest

import rewards
from rewards import execution_reward_func, process_reward_func, strict_format_reward_func


class RewardTests(unittest.TestCase):
    def setUp(self):
        rewards.AERO_GLOBAL_STEP = 0
        rewards.AERO_MAX_STEPS = 100

    def test_strict_format_penalty_without_answer_tag(self):
        prompts = ["irrelevant"]
        completions = [{"content": "plain text without tags"}]
        values = strict_format_reward_func(prompts=prompts, completions=completions)
        self.assertEqual(len(values), 1)
        self.assertLess(values[0], 0.0)

    def test_execution_reward_partial_signal(self):
        prompts = ["solve"]
        completions = [{"content": "<answer>\ndef add(a, b):\n    return a + b\n</answer>"}]
        tests = ["assert add(1, 2) == 3\nassert add(2, 2) == 5"]
        values = execution_reward_func(prompts=prompts, completions=completions, answer=tests)
        self.assertEqual(len(values), 1)
        self.assertGreater(values[0], -3.0)
        self.assertLess(values[0], 2.5)

    def test_execution_reward_success(self):
        prompts = ["solve"]
        completions = [{"content": "<answer>\ndef mul(a, b):\n    return a * b\n</answer>"}]
        tests = ["assert mul(2, 3) == 6\nassert mul(0, 5) == 0"]
        values = execution_reward_func(prompts=prompts, completions=completions, answer=tests)
        self.assertEqual(len(values), 1)
        self.assertGreater(values[0], 1.0)

    def test_process_reward_prefers_failure_aligned_reasoning(self):
        prompts = ["task ... error_type=timeout ..."]
        tests = ["assert solve(1) == 1"]
        strong = [
            {
                "content": (
                    "<reasoning>First optimize complexity and handle boundary cases."
                    " Then validate with asserts for regression tests.</reasoning>"
                    "<answer>def solve(x):\n    return x\n</answer>"
                )
            }
        ]
        weak = [{"content": "<reasoning>quick try</reasoning><answer>x</answer>"}]
        strong_score = process_reward_func(prompts=prompts, completions=strong, answer=tests)[0]
        weak_score = process_reward_func(prompts=prompts, completions=weak, answer=tests)[0]
        self.assertGreater(strong_score, weak_score)

    def test_diversity_reward_penalizes_duplicates(self):
        prompts = ["same prompt", "same prompt"]
        completions = [
            {"content": "<answer>def f(x):\n    return x + 1\n</answer>"},
            {"content": "<answer>def f(x):\n    return x + 1\n</answer>"},
        ]
        values = rewards.diversity_exploration_reward_func(
            prompts=prompts,
            completions=completions,
            answer=["", ""],
        )
        self.assertEqual(len(values), 2)
        self.assertGreater(values[0], values[1])


if __name__ == "__main__":
    unittest.main()
