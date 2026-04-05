import unittest
from knn_cli.statistics import (
    mean_dataset,
    median_dataset,
    quartile_values_dataset,
    count_min_max,
    standard_deviation_dataset
)

class TestMeanDataset(unittest.TestCase):

    def test_mean_single_feature(self):
        col = {"a": [1.0, 2.0, 3.0]}
        result = mean_dataset(col)
        self.assertAlmostEqual(result["a"], 2.0, places=2)

    def test_mean_multiple_features(self):
        col = {"a": [1.0, 3.0], "b": [2.0, 4.0]}
        result = mean_dataset(col)
        self.assertAlmostEqual(result["a"], 2.0, places=2)
        self.assertAlmostEqual(result["b"], 3.0, places=2)

    def test_mean_single_value(self):
        col = {"a": [5.0]}
        result = mean_dataset(col)
        self.assertAlmostEqual(result["a"], 5.0, places=2)

    def test_mean_negative_values(self):
        col = {"a": [-3.0, -1.0, -2.0]}
        result = mean_dataset(col)
        self.assertAlmostEqual(result["a"], -2.0, places=2)

    def test_mean_returns_all_features(self):
        col = {"a": [1.0], "b": [2.0], "c": [3.0]}
        result = mean_dataset(col)
        self.assertEqual(set(result.keys()), {"a", "b", "c"})


class TestMedianDataset(unittest.TestCase):

    def test_median_odd_length(self):
        col = {"a": [1.0, 2.0, 3.0]}
        result = median_dataset(col)
        self.assertAlmostEqual(result["a"], 2.0, places=5)

    def test_median_even_length(self):
        col = {"a": [1.0, 2.0, 3.0, 4.0]}
        result = median_dataset(col)
        self.assertAlmostEqual(result["a"], 2.5, places=5)

    def test_median_unsorted_input(self):
        col = {"a": [5.0, 1.0, 3.0]}
        result = median_dataset(col)
        self.assertAlmostEqual(result["a"], 3.0, places=5)

    def test_median_single_value(self):
        col = {"a": [7.0]}
        result = median_dataset(col)
        self.assertAlmostEqual(result["a"], 7.0, places=5)

    def test_median_multiple_features(self):
        col = {"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]}
        result = median_dataset(col)
        self.assertAlmostEqual(result["a"], 2.0, places=5)
        self.assertAlmostEqual(result["b"], 20.0, places=5)


class TestCountMinMax(unittest.TestCase):

    def test_count_is_correct(self):
        col = {"a": [1.0, 2.0, 3.0]}
        count, _, _ = count_min_max(col)
        self.assertEqual(count, 3)

    def test_min_values(self):
        col = {"a": [3.0, 1.0, 2.0], "b": [10.0, 5.0, 8.0]}
        _, min_map, _ = count_min_max(col)
        self.assertAlmostEqual(min_map["a"], 1.0, places=5)
        self.assertAlmostEqual(min_map["b"], 5.0, places=5)

    def test_max_values(self):
        col = {"a": [3.0, 1.0, 2.0], "b": [10.0, 5.0, 8.0]}
        _, _, max_map = count_min_max(col)
        self.assertAlmostEqual(max_map["a"], 3.0, places=5)
        self.assertAlmostEqual(max_map["b"], 10.0, places=5)

    def test_single_value_min_equals_max(self):
        col = {"a": [4.0]}
        _, min_map, max_map = count_min_max(col)
        self.assertAlmostEqual(min_map["a"], max_map["a"], places=5)

    def test_negative_values(self):
        col = {"a": [-5.0, -1.0, -3.0]}
        _, min_map, max_map = count_min_max(col)
        self.assertAlmostEqual(min_map["a"], -5.0, places=5)
        self.assertAlmostEqual(max_map["a"], -1.0, places=5)


class TestStandardDeviationDataset(unittest.TestCase):

    def test_std_known_value(self):
        col = {"a": [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]}
        result = standard_deviation_dataset(col)
        self.assertAlmostEqual(result["a"], 2.0, places=2)

    def test_std_constant_column_is_zero(self):
        col = {"a": [5.0, 5.0, 5.0]}
        result = standard_deviation_dataset(col)
        self.assertAlmostEqual(result["a"], 0.0, places=5)

    def test_std_single_value_is_zero(self):
        col = {"a": [3.0]}
        result = standard_deviation_dataset(col)
        self.assertAlmostEqual(result["a"], 0.0, places=5)

    def test_std_multiple_features(self):
        col = {"a": [1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0]}
        result = standard_deviation_dataset(col)
        self.assertAlmostEqual(result["a"], 0.0, places=5)
        self.assertGreater(result["b"], 0.0)

    def test_std_is_nonnegative(self):
        col = {"a": [1.0, 5.0, 3.0, 9.0, 2.0]}
        result = standard_deviation_dataset(col)
        self.assertGreaterEqual(result["a"], 0.0)


class TestQuartileValuesDataset(unittest.TestCase):

    def test_q1_and_q3_basic(self):
        col = {"a": [1.0, 2.0, 3.0, 4.0, 5.0]}
        q1, q3 = quartile_values_dataset(col)
        self.assertLess(q1["a"], 3.0)
        self.assertGreater(q3["a"], 3.0)

    def test_q1_less_than_q3(self):
        col = {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
        q1, q3 = quartile_values_dataset(col)
        self.assertLess(q1["a"], q3["a"])

    def test_quartiles_multiple_features(self):
        col = {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0]
        }
        q1, q3 = quartile_values_dataset(col)
        self.assertIn("a", q1)
        self.assertIn("b", q1)
        self.assertIn("a", q3)
        self.assertIn("b", q3)


if __name__ == "__main__":
    unittest.main()