import unittest

from prm_tiny import predict_tiny_prm, train_tiny_prm


class TinyPRMTests(unittest.TestCase):
    def test_tiny_prm_learns_signal(self):
        rows = [
            ("prompt=fix bug timeout optimize complexity", 1),
            ("prompt=clean patch regression test pass", 1),
            ("prompt=random broken syntax crash", 0),
            ("prompt=malformed answer missing tags", 0),
        ]
        model = train_tiny_prm(rows, buckets=1024, epochs=6, learning_rate=0.1)
        pos = predict_tiny_prm(model, "optimize complexity and regression test")
        neg = predict_tiny_prm(model, "broken syntax malformed output")
        self.assertGreater(pos, neg)


if __name__ == "__main__":
    unittest.main()
