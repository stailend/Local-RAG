import unittest

import numpy as np

from experiments.hierarchical_retrieval import Hierarchy, normalize


class HierarchyTest(unittest.TestCase):
    def test_finds_clustered_vectors_with_fewer_comparisons(self):
        random = np.random.default_rng(7)
        centers = normalize(random.normal(size=(8, 16)))
        vectors = normalize(np.vstack([
            center + random.normal(scale=0.01, size=(16, 16)) for center in centers
        ]))
        tree = Hierarchy.build(vectors, branching=4, method="centroid")

        found, comparisons = tree.search(vectors[37], top_k=5, beam=2)

        self.assertIn(37, found)
        self.assertLess(comparisons, len(vectors))


if __name__ == "__main__":
    unittest.main()
