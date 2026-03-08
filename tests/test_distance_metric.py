import unittest
from dataclasses import dataclass
from knn_cli.distance_metric import euclidean, manhattan, cosine

@dataclass
class TestPoints:
    p1: list[float]
    p2: list[float]

class TestDistanceMetric(unittest.TestCase):

    def setUp(self):
        self.data = [
            TestPoints([1,2], [4,6]),
            TestPoints([1,2,3], [4,6,3]),
            TestPoints([1,2,3,4], [1,2,3,10])
        ]

    def test_euclidean_distance(self):
        expected = [5.0, 5.0, 6.0]
        for idx, points in enumerate(self.data):
            result = euclidean(points.p1, points.p2)
            self.assertAlmostEqual(result, expected[idx], places=5)

    def test_manhattan_distance(self):
        expected = [7, 7, 6]
        for idx, points in enumerate(self.data):
            result = manhattan(points.p1, points.p2)
            self.assertEqual(result, expected[idx])

    def test_cosine_distance(self):
        data = self.data + [
            TestPoints([1,0], [0,1]),
            TestPoints([0,1], [0,-1])
        ]

        expected = [0.007722, 0.144517, 0.0766194, 1.0, 2.0]

        for idx, points in enumerate(data):
            result = cosine(points.p1, points.p2)
            self.assertAlmostEqual(result, expected[idx], places=5)

    def test_zero_distance(self):
        p = self.data[0].p1

        self.assertEqual(euclidean(p, p), 0.0)
        self.assertEqual(manhattan(p, p), 0.0)
        self.assertAlmostEqual(cosine(p, p), 0.0, places=5)

    def test_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            euclidean([1, 2, 3], [1, 2])

if __name__ == "__main__":
    unittest.main()