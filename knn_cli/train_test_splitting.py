from collections import Counter

from knn_cli.data_utils import Datapoint
from knn_cli.knn import calculate_distances, k_nearest_points, get_classification

from random import shuffle
from copy import deepcopy
from math import floor

def train_test_split(datapoints: list[Datapoint], fraction: float) -> tuple[list[Datapoint], list[Datapoint]]:
    """
    Splits a dataset into training and testing sets based on the given fraction.
    The dataset is shuffled before splitting to ensure randomness.

    Raises ValueError if the testing set would be empty, or if the training
    set would have fewer than 2 datapoints.

    :param datapoints: list of Datapoint objects to split.
    :param fraction: the proportion of data to use for testing. Must be between 0 and 1 exclusive.

    :return: a tuple of (training, testing) lists of Datapoint objects.
    """
    copied_data = deepcopy(datapoints)
    shuffle(copied_data)

    testing_size = floor(fraction * len(copied_data))
    if testing_size == 0:
        raise ValueError("Fraction is too small for conducting a meaningful train-test split.")

    if testing_size >= len(copied_data):
        raise ValueError("Testing split fraction is too large and leaves no meaningful training data.")

    training_size = len(copied_data) - testing_size
    if training_size < 2:
        raise ValueError("Testing split fraction is too large and leaves no meaningful training data.")

    testing_data = copied_data[training_size:]
    training_data = copied_data[:training_size]

    return training_data, testing_data

def get_accuracy(k: int, distance: str, training_data: list[Datapoint], testing_data: list[Datapoint]) -> float:
    """
    Evaluates the KNN model's classification accuracy on the testing set.

    For each point in the testing set, runs the full KNN pipeline against the
    training data and compares the predicted category to the actual category.

    :param k: number of nearest neighbors to consider.
    :param distance: the distance metric to use. Accepts 'eucl', 'manh', or 'cos'.
    :param training_data: list of Datapoint objects used for training.
    :param testing_data: list of Datapoint objects used for evaluation.

    :return: fraction of correctly classified testing points as a float in the closed interval [0, 1].
    """
    correct_preds = 0
    for test_point in testing_data:
        distances = calculate_distances(list(test_point.features), training_data, distance)
        nearest_neighbors = k_nearest_points(k, distances)
        algo_prediction = get_classification(nearest_neighbors)

        if algo_prediction == test_point.category: #found accurate classification
            correct_preds += 1

    return correct_preds / len(testing_data)


def get_baseline_accuracy(training: list[Datapoint], testing: list[Datapoint]):
    """
    Computes the baseline accuracy of a naive classifier that always predicts
    the most frequent category found in the training set.

    :param training: list of Datapoint objects used to determine the majority class.
    :param testing: list of Datapoint objects to evaluate the baseline against.

    :return: fraction of testing points whose category matches the majority
    training class, as a float in the closed interval [0, 1].
    """
    counter = Counter([pt.category for pt in training])
    most_frequent_category = counter.most_common(1)[0][0]

    req_count = 0
    #for each point in testing data, if its category is the most frequent category (mfc), increment count
    for point in testing:
        if point.category == most_frequent_category:
            req_count += 1

    return req_count / len(testing)





