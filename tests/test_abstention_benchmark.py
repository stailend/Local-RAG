import unittest

import numpy as np

from experiments.abstention_benchmark import choose_threshold, evaluate


class AbstentionBenchmarkTest(unittest.TestCase):
    def test_threshold_measures_both_error_types(self):
        scores = np.array([[0.8, 0.2], [0.6, 0.7]])
        self.assertEqual(evaluate(scores, 0.5)["false_accept_rate"], 0.5)
        threshold = choose_threshold(scores, max_false_accept=0)
        self.assertGreater(threshold, 0.7)
        self.assertEqual(evaluate(scores, threshold)["false_accept_rate"], 0)


if __name__ == "__main__":
    unittest.main()
