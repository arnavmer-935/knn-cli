"""
    Unit test cases to verify correctness of the distance_metric.py algorithms.

    This module contains test for:
    - Euclidean distance
    - Manhattan distance
    - Cosine distance
"""
import unittest
from distance_metric import euclidean, manhattan, cosine
from dataclasses import dataclass


@dataclass
class TestPoints:
    """
    Class to represent a test case for distance metrics
    p1: The first n-dimensional data point
    p2: The second n-dimension data point
    """
    p1: list[float]
    p2: list[float]


class TestDistanceMetric(unittest.TestCase):
    """
    Unit class to verify the correctness of the distance metric algorithms.
    """

    __slots__ = "_data"
    _data: list[TestPoints]

    def setUp(self):
        """
        This method is called before calling any test method.
        """
        test1 = TestPoints([1, 2], [4, 6])
        test2 = TestPoints([1, 2, 3], [4, 6, 3])
        test3 = TestPoints([1, 2, 3, 4], [1, 2, 3, 10])
        self._data = [test1, test2, test3]

    def test_euclidean_distance(self):
        """
        Unit test to verify the Euclidean distance algorithm
        :return: None
        """
        expected = [5.0, 5.0, 6.0]
        for idx, test_points in enumerate(self._data):
            result = euclidean(test_points.p1, test_points.p2)
            self.assertAlmostEqual(result, expected[idx], places=5)

    def test_manhattan_distance(self):
        """
        Unit test to verify the Manhattan distance algorithm
        :return: None
        """
        expected = [7, 7, 6]
        for idx, test_points in enumerate(self._data):
            result = manhattan(test_points.p1, test_points.p2)
            self.assertEqual(result, expected[idx])

    def test_cosine(self):
        self._data.append(TestPoints([1, 0], [0, 1]))  # orthogonal vectors, expected 1.0
        self._data.append(TestPoints([0, 1], [0, -1]))  # opposite vectors, expected 2.0

        expected = [0.007722, 0.144517, 0.0766194, 1.0, 2.0]
        for idx, test_points in enumerate(self._data):
            result = cosine(test_points.p1, test_points.p2)
            self.assertAlmostEqual(result, expected[idx], 5)

    def test_zero_distance(self):
        """
        Test all distance metrics using the same point.
        The distance should be 0 for all distance metrics.
        :return: None
        """
        test_points = self._data[0]
        self.assertEqual(euclidean(test_points.p1, test_points.p1), 0.0)
        self.assertEqual(manhattan(test_points.p1, test_points.p1), 0.0)
        self.assertAlmostEqual(cosine(test_points.p1, test_points.p1), 0.0, 5)


if __name__ == '__main__':
    unittest.main()