from knn_cli.data_utils import Datapoint
from knn_cli.knn import calculate_distances, k_nearest_points, get_classification

from random import shuffle
from copy import deepcopy
from math import floor

def train_test_split(datapoints: list[Datapoint], fraction: float) -> tuple[list[Datapoint], list[Datapoint]]:

    copied_data = deepcopy(datapoints)
    shuffle(copied_data)

    testing_size = floor(fraction * len(copied_data))
    if testing_size == 0:
        raise ValueError("Fraction is too small for conducting a meaningful train-test split.")

    if testing_size == len(copied_data):
        raise ValueError("Testing split fraction is too large and leaves no meaningful training data.")

    training_size = len(copied_data) - testing_size
    testing_data = copied_data[training_size:]
    training_data = copied_data[:training_size]

    return training_data, testing_data

def get_accuracy(k: int, distance: str, training_data: list[Datapoint], testing_data: list[Datapoint]) -> float:

    correct_preds = 0
    for test_point in testing_data:
        distances = calculate_distances(list(test_point.features), training_data, distance)
        nearest_neighbors = k_nearest_points(k, distances)
        algo_prediction = get_classification(nearest_neighbors)

        if algo_prediction == test_point.category: #found accurate classification
            correct_preds += 1

    return correct_preds / len(testing_data)




