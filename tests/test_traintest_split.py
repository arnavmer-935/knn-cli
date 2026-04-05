import unittest
from knn_cli.data_utils import Datapoint
from knn_cli.train_test_splitting import train_test_split, get_accuracy, get_baseline_accuracy


class TestTrainTestSplit(unittest.TestCase):

    def setUp(self):
        self.datapoints = [Datapoint((float(i), float(i)), "A" if i < 5 else "B")
                           for i in range(10)]

    def test_split_sizes_are_correct(self):
        training, testing = train_test_split(self.datapoints, 0.2)
        self.assertEqual(len(training), 8)
        self.assertEqual(len(testing), 2)

    def test_no_datapoints_lost_in_split(self):
        training, testing = train_test_split(self.datapoints, 0.3)
        self.assertEqual(len(training) + len(testing), len(self.datapoints))

    def test_no_overlap_between_splits(self):
        training, testing = train_test_split(self.datapoints, 0.2)
        training_ids = {id(pt) for pt in training}
        testing_ids = {id(pt) for pt in testing}
        self.assertTrue(training_ids.isdisjoint(testing_ids))

    def test_original_data_is_not_modified(self):
        original_first = self.datapoints[0]
        train_test_split(self.datapoints, 0.2)
        self.assertEqual(self.datapoints[0], original_first)

    def test_fraction_too_small_raises_value_error(self):
        tiny_data = [Datapoint((1.0,), "A"), Datapoint((2.0,), "B")]
        with self.assertRaises(ValueError):
            train_test_split(tiny_data, 0.01)

    def test_fraction_too_large_raises_value_error(self):
        small_data = [Datapoint((float(i),), "A") for i in range(3)]
        with self.assertRaises(ValueError):
            train_test_split(small_data, 0.8)


class TestGetAccuracy(unittest.TestCase):

    def setUp(self):
        self.training = [
            Datapoint((1.0, 1.0), "A"),
            Datapoint((1.1, 1.1), "A"),
            Datapoint((1.2, 1.0), "A"),
            Datapoint((9.0, 9.0), "B"),
            Datapoint((9.1, 8.9), "B"),
            Datapoint((8.9, 9.1), "B"),
        ]
        self.testing = [
            Datapoint((1.05, 1.05), "A"),
            Datapoint((9.05, 9.05), "B"),
        ]

    def test_perfect_accuracy_on_well_separated_data(self):
        accuracy = get_accuracy(3, "eucl", self.training, self.testing)
        self.assertAlmostEqual(accuracy, 1.0, places=5)

    def test_accuracy_is_between_zero_and_one(self):
        accuracy = get_accuracy(3, "eucl", self.training, self.testing)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_accuracy_works_with_manhattan(self):
        accuracy = get_accuracy(3, "manh", self.training, self.testing)
        self.assertAlmostEqual(accuracy, 1.0, places=5)

    def test_accuracy_works_with_cosine(self):
        accuracy = get_accuracy(1, "cos", self.training, self.testing)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)


class TestGetBaselineAccuracy(unittest.TestCase):

    def test_baseline_returns_majority_class_fraction(self):
        training = [
            Datapoint((1.0,), "A"),
            Datapoint((2.0,), "A"),
            Datapoint((3.0,), "A"),
            Datapoint((4.0,), "B"),
        ]
        testing = [
            Datapoint((5.0,), "A"),
            Datapoint((6.0,), "A"),
            Datapoint((7.0,), "B"),
        ]
        # majority class in training is "A", 2 of 3 test points are "A"
        baseline = get_baseline_accuracy(training, testing)
        self.assertAlmostEqual(baseline, 2/3, places=5)

    def test_baseline_is_between_zero_and_one(self):
        training = [Datapoint((1.0,), "A"), Datapoint((2.0,), "B")]
        testing = [Datapoint((3.0,), "A"), Datapoint((4.0,), "B")]
        baseline = get_baseline_accuracy(training, testing)
        self.assertGreaterEqual(baseline, 0.0)
        self.assertLessEqual(baseline, 1.0)

    def test_baseline_all_same_class(self):
        training = [Datapoint((float(i),), "A") for i in range(5)]
        testing = [Datapoint((float(i),), "A") for i in range(3)]
        baseline = get_baseline_accuracy(training, testing)
        self.assertAlmostEqual(baseline, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()