import unittest
from knn_cli.data_utils import Datapoint
from knn_cli.normalization import (
    minmax,
    zscore,
    normalized_values_minmax,
    normalized_values_zscore,
    get_normalized_datapoints,
    get_min_max_map,
    get_mean_std_map,
)


class TestMinMax(unittest.TestCase):

    def test_known_input_produces_correct_output(self):
        result = minmax([2.0, 5.0, 10.0], 0.0, 10.0)
        self.assertAlmostEqual(result[0], 0.2, places=5)
        self.assertAlmostEqual(result[1], 0.5, places=5)
        self.assertAlmostEqual(result[2], 1.0, places=5)

    def test_minimum_value_scales_to_zero(self):
        result = minmax([0.0, 5.0, 10.0], 0.0, 10.0)
        self.assertAlmostEqual(result[0], 0.0, places=5)

    def test_maximum_value_scales_to_one(self):
        result = minmax([0.0, 5.0, 10.0], 0.0, 10.0)
        self.assertAlmostEqual(result[2], 1.0, places=5)

    def test_min_equals_max_returns_all_zeros(self):
        result = minmax([5.0, 5.0, 5.0], 5.0, 5.0)
        self.assertEqual(result, [0, 0, 0])

    def test_single_value(self):
        result = minmax([3.0], 0.0, 10.0)
        self.assertAlmostEqual(result[0], 0.3, places=5)

    def test_negative_values_scale_correctly(self):
        result = minmax([-1.0], -3.0, 3.0)
        self.assertAlmostEqual(result[0], 1/3, places=5)


class TestZScore(unittest.TestCase):

    def test_known_input_produces_correct_output(self):
        result = zscore([1.0, 2.0, 3.0], 2.0, 1.0)
        self.assertAlmostEqual(result[0], -1.0, places=5)
        self.assertAlmostEqual(result[1],  0.0, places=5)
        self.assertAlmostEqual(result[2],  1.0, places=5)

    def test_zero_std_returns_all_zeros(self):
        result = zscore([5.0, 5.0, 5.0], 5.0, 0.0)
        self.assertEqual(result, [0, 0, 0])

    def test_mean_of_normalized_output_is_approximately_zero(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        result = zscore(values, mean, std)
        self.assertAlmostEqual(sum(result) / len(result), 0.0, places=5)

    def test_std_of_normalized_output_is_approximately_one(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        result = zscore(values, mean, std)
        result_mean = sum(result) / len(result)
        result_std = (sum((v - result_mean) ** 2 for v in result) / len(result)) ** 0.5
        self.assertAlmostEqual(result_std, 1.0, places=5)

    def test_single_value(self):
        result = zscore([3.0], 2.0, 1.0)
        self.assertAlmostEqual(result[0], 1.0, places=5)

    def test_negative_values_scale_correctly(self):
        # (-3 - 0) / 1 = -3.0
        result = zscore([-3.0], 0.0, 1.0)
        self.assertAlmostEqual(result[0], -3.0, places=5)


class TestNormalizedValuesMinMax(unittest.TestCase):

    def setUp(self):
        self.datapoints = [
            Datapoint((1.0, 10.0), "A"),
            Datapoint((2.0, 20.0), "B"),
            Datapoint((3.0, 30.0), "A"),
        ]
        self.feature_map = {"col1": 0, "col2": 1}
        self.min_max_map = {"col1": (1.0, 3.0), "col2": (10.0, 30.0)}
        self.query_point = [2.0, 20.0]

    def test_output_contains_same_keys_as_feature_map(self):
        result, _ = normalized_values_minmax(
            self.datapoints, self.feature_map, self.min_max_map, self.query_point
        )
        self.assertEqual(set(result.keys()), set(self.feature_map.keys()))

    def test_values_bounded_between_zero_and_one(self):
        result, _ = normalized_values_minmax(
            self.datapoints, self.feature_map, self.min_max_map, self.query_point
        )
        for col, values in result.items():
            for v in values:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_known_column_produces_correct_values(self):
        # col1: (1-1)/(3-1)=0.0, (2-1)/(3-1)=0.5, (3-1)/(3-1)=1.0
        result, _ = normalized_values_minmax(
            self.datapoints, self.feature_map, self.min_max_map, self.query_point
        )
        self.assertAlmostEqual(result["col1"][0], 0.0, places=5)
        self.assertAlmostEqual(result["col1"][1], 0.5, places=5)
        self.assertAlmostEqual(result["col1"][2], 1.0, places=5)

    def test_query_point_normalized_correctly(self):
        # col1: (2-1)/(3-1) = 0.5, col2: (20-10)/(30-10) = 0.5
        _, normalized_pt = normalized_values_minmax(
            self.datapoints, self.feature_map, self.min_max_map, self.query_point
        )
        self.assertAlmostEqual(normalized_pt[0], 0.5, places=5)
        self.assertAlmostEqual(normalized_pt[1], 0.5, places=5)


class TestNormalizedValuesZScore(unittest.TestCase):

    def setUp(self):
        self.datapoints = [
            Datapoint((1.0, 10.0), "A"),
            Datapoint((2.0, 20.0), "B"),
            Datapoint((3.0, 30.0), "A"),
        ]
        self.feature_map = {"col1": 0, "col2": 1}
        self.mean_std_map = {"col1": (2.0, 1.0), "col2": (20.0, 10.0)}
        self.query_point = [2.0, 20.0]

    def test_output_contains_same_keys_as_feature_map(self):
        result, _ = normalized_values_zscore(
            self.datapoints, self.feature_map, self.mean_std_map, self.query_point
        )
        self.assertEqual(set(result.keys()), set(self.feature_map.keys()))

    def test_known_column_produces_correct_values(self):
        # col1: (1-2)/1=-1, (2-2)/1=0, (3-2)/1=1
        result, _ = normalized_values_zscore(
            self.datapoints, self.feature_map, self.mean_std_map, self.query_point
        )
        self.assertAlmostEqual(result["col1"][0], -1.0, places=5)
        self.assertAlmostEqual(result["col1"][1],  0.0, places=5)
        self.assertAlmostEqual(result["col1"][2],  1.0, places=5)

    def test_zero_std_column_returns_all_zeros(self):
        mean_std_map = {"col1": (2.0, 0.0), "col2": (20.0, 10.0)}
        result, _ = normalized_values_zscore(
            self.datapoints, self.feature_map, mean_std_map, self.query_point
        )
        self.assertEqual(result["col1"], [0, 0, 0])

    def test_query_point_normalized_correctly(self):
        # col1: (2-2)/1=0.0, col2: (20-20)/10=0.0
        _, normalized_pt = normalized_values_zscore(
            self.datapoints, self.feature_map, self.mean_std_map, self.query_point
        )
        self.assertAlmostEqual(normalized_pt[0], 0.0, places=5)
        self.assertAlmostEqual(normalized_pt[1], 0.0, places=5)


class TestGetNormalizedDatapoints(unittest.TestCase):

    def setUp(self):
        self.datapoints = [
            Datapoint((1.0, 10.0), "A"),
            Datapoint((2.0, 20.0), "B"),
            Datapoint((3.0, 30.0), "A"),
        ]
        self.feature_map = {"col1": 0, "col2": 1}
        self.normalized_values = {
            "col1": [0.0, 0.5, 1.0],
            "col2": [0.0, 0.5, 1.0],
        }

    def test_returns_same_number_of_datapoints(self):
        result = get_normalized_datapoints(
            self.datapoints, self.normalized_values, self.feature_map
        )
        self.assertEqual(len(result), len(self.datapoints))

    def test_category_labels_are_preserved(self):
        result = get_normalized_datapoints(
            self.datapoints, self.normalized_values, self.feature_map
        )
        for i, pt in enumerate(result):
            self.assertEqual(pt.category, self.datapoints[i].category)

    def test_feature_values_match_normalized_column_values(self):
        result = get_normalized_datapoints(
            self.datapoints, self.normalized_values, self.feature_map
        )
        for i, pt in enumerate(result):
            self.assertAlmostEqual(pt.features[0], self.normalized_values["col1"][i], places=5)
            self.assertAlmostEqual(pt.features[1], self.normalized_values["col2"][i], places=5)

    def test_returns_datapoint_objects(self):
        result = get_normalized_datapoints(
            self.datapoints, self.normalized_values, self.feature_map
        )
        for pt in result:
            self.assertIsInstance(pt, Datapoint)


class TestGetMinMaxMap(unittest.TestCase):

    def test_correctly_pairs_min_and_max(self):
        min_map = {"col1": 1.0, "col2": 10.0}
        max_map = {"col1": 3.0, "col2": 30.0}
        result = get_min_max_map(min_map, max_map)
        self.assertEqual(result["col1"], (1.0, 3.0))
        self.assertEqual(result["col2"], (10.0, 30.0))

    def test_output_contains_same_keys(self):
        min_map = {"col1": 1.0, "col2": 10.0}
        max_map = {"col1": 3.0, "col2": 30.0}
        result = get_min_max_map(min_map, max_map)
        self.assertEqual(set(result.keys()), {"col1", "col2"})


class TestGetMeanStdMap(unittest.TestCase):

    def test_correctly_pairs_mean_and_std(self):
        mean_map = {"col1": 2.0, "col2": 20.0}
        std_map  = {"col1": 1.0, "col2": 10.0}
        result = get_mean_std_map(mean_map, std_map)
        self.assertEqual(result["col1"], (2.0, 1.0))
        self.assertEqual(result["col2"], (20.0, 10.0))

    def test_output_contains_same_keys(self):
        mean_map = {"col1": 2.0, "col2": 20.0}
        std_map  = {"col1": 1.0, "col2": 10.0}
        result = get_mean_std_map(mean_map, std_map)
        self.assertEqual(set(result.keys()), {"col1", "col2"})


if __name__ == "__main__":
    unittest.main()
