import unittest
from knn_cli.data_utils import Datapoint
from knn_cli.knn import calculate_distances, k_nearest_points, get_classification
from knn_cli.distance_metric import cosine

class TestCalculateDistances(unittest.TestCase):

    def setUp(self):
        self.datapoints = [
            Datapoint((1.0, 2.0), "A"),
            Datapoint((3.0, 4.0), "B"),
            Datapoint((5.0, 6.0), "A"),
            Datapoint((7.0, 8.0), "B"),
            Datapoint((2.0, 3.0), "A"),
        ]
        self.query = [1.0, 2.0]

    def test_returns_one_entry_per_datapoint(self):
        result = calculate_distances(self.query, self.datapoints)
        self.assertEqual(len(result), len(self.datapoints))

    def test_euclidean_distance_to_identical_point_is_zero(self):
        result = calculate_distances(self.query, self.datapoints, "eucl")
        distances = [d for d, _ in result]
        self.assertAlmostEqual(distances[0], 0.0, places=5)

    def test_euclidean_known_distance(self):
        result = calculate_distances(self.query, self.datapoints, "eucl")
        distances = [d for d, _ in result]
        self.assertAlmostEqual(distances[1], 2.8284271, places=5)

    def test_manhattan_distance_to_identical_point_is_zero(self):
        result = calculate_distances(self.query, self.datapoints, "manh")
        distances = [d for d, _ in result]
        self.assertAlmostEqual(distances[0], 0.0, places=5)

    def test_manhattan_known_distance(self):
        result = calculate_distances(self.query, self.datapoints, "manh")
        distances = [d for d, _ in result]
        self.assertAlmostEqual(distances[1], 4.0, places=5)

    def test_cosine_distance_to_identical_point_is_zero(self):
        result = calculate_distances(self.query, self.datapoints, "cos")
        distances = [d for d, _ in result]
        self.assertAlmostEqual(distances[0], 0.0, places=5)

    def test_unknown_metric_defaults_to_euclidean(self):
        result_default = calculate_distances(self.query, self.datapoints)
        result_eucl = calculate_distances(self.query, self.datapoints, "eucl")
        for (d1, _), (d2, _) in zip(result_default, result_eucl):
            self.assertAlmostEqual(d1, d2, places=5)

    def test_result_contains_original_datapoint_references(self):
        result = calculate_distances(self.query, self.datapoints, "eucl")
        returned_points = [pt for _, pt in result]
        for point in self.datapoints:
            self.assertIn(point, returned_points)

class TestKNearestPoints(unittest.TestCase):

    def setUp(self):
        self.datapoints = [
            Datapoint((1.0, 2.0), "A"),
            Datapoint((3.0, 4.0), "B"),
            Datapoint((5.0, 6.0), "A"),
            Datapoint((7.0, 8.0), "B"),
            Datapoint((2.0, 3.0), "A"),
        ]
        self.query = [1.0, 2.0]
        self.distances = calculate_distances(self.query, self.datapoints, "eucl")

    def test_returns_exactly_k_neighbors(self):
        for k in [1, 2, 3]:
            result = k_nearest_points(k, self.distances)
            self.assertEqual(len(result), k)

    def test_k_equals_1_returns_closest_point(self):
        result = k_nearest_points(1, self.distances)
        # query is identical to the first datapoint, so it should be returned
        self.assertEqual(result[0], self.datapoints[0])

    def test_k_equals_dataset_size_returns_all_points(self):
        result = k_nearest_points(len(self.datapoints), self.distances)
        self.assertEqual(len(result), len(self.datapoints))
        for point in self.datapoints:
            self.assertIn(point, result)

    def test_neighbors_are_ordered_closest_first(self):
        result = k_nearest_points(3, self.distances)
        dist_map = {id(pt): d for d, pt in self.distances}
        returned_distances = [dist_map[id(pt)] for pt in result]
        self.assertEqual(returned_distances, sorted(returned_distances))

    def test_does_not_modify_input_distances(self):
        original_order = [(d, pt) for d, pt in self.distances]
        k_nearest_points(3, self.distances)
        for i, (d, pt) in enumerate(self.distances):
            self.assertAlmostEqual(d, original_order[i][0], places=5)
            self.assertEqual(pt, original_order[i][1])


class TestGetClassification(unittest.TestCase):

    def test_clear_majority_vote(self):
        neighbors = [
            Datapoint((1.0, 2.0), "A"),
            Datapoint((2.0, 3.0), "A"),
            Datapoint((3.0, 4.0), "B"),
            Datapoint((4.0, 5.0), "A"),
        ]
        self.assertEqual(get_classification(neighbors), "A")

    def test_single_neighbor_returns_its_class(self):
        neighbors = [Datapoint((1.0, 2.0), "B")]
        self.assertEqual(get_classification(neighbors), "B")

    def test_all_same_class(self):
        neighbors = [
            Datapoint((1.0, 2.0), "A"),
            Datapoint((3.0, 4.0), "A"),
            Datapoint((5.0, 6.0), "A"),
        ]
        self.assertEqual(get_classification(neighbors), "A")

    def test_tie_breaking_returns_first_encountered_class(self):
        neighbors = [
            Datapoint((1.0, 2.0), "A"),
            Datapoint((2.0, 3.0), "B"),
        ]
        self.assertEqual(get_classification(neighbors), "A")

    def test_three_way_tie_returns_first_encountered_class(self):
        neighbors = [
            Datapoint((1.0, 2.0), "C"),
            Datapoint((2.0, 3.0), "A"),
            Datapoint((3.0, 4.0), "B"),
        ]
        self.assertEqual(get_classification(neighbors), "C")


class TestFullKNNPipeline(unittest.TestCase):

    def setUp(self):
        self.training = [
            Datapoint((1.0, 1.0), "A"),
            Datapoint((1.2, 1.1), "A"),
            Datapoint((1.1, 1.3), "A"),
            Datapoint((9.0, 9.0), "B"),
            Datapoint((9.2, 8.9), "B"),
            Datapoint((8.8, 9.1), "B"),
        ]

    def test_query_near_class_a_predicts_a(self):
        query = [1.05, 1.05]
        distances = calculate_distances(query, self.training, "eucl")
        neighbors = k_nearest_points(3, distances)
        self.assertEqual(get_classification(neighbors), "A")

    def test_query_near_class_b_predicts_b(self):
        query = [9.1, 9.0]
        distances = calculate_distances(query, self.training, "eucl")
        neighbors = k_nearest_points(3, distances)
        self.assertEqual(get_classification(neighbors), "B")

    def test_pipeline_consistent_across_distance_metrics(self):
        query = [1.05, 1.05]
        for metric in ["eucl", "manh"]:
            distances = calculate_distances(query, self.training, metric)
            neighbors = k_nearest_points(3, distances)
            prediction = get_classification(neighbors)
            self.assertEqual(prediction, "A", msg=f"Failed with metric: {metric}")

    def test_cosine_uses_angle_not_magnitude(self):
        self.assertAlmostEqual(cosine([1.0, 1.0], [9.0, 9.0]), 0.0, places=5)

if __name__ == "__main__":
    unittest.main()