from knn_cli.data_utils import Datapoint
from knn_cli.distance_metric import euclidean, manhattan, cosine
from collections import Counter

def calculate_distances(feature: list[float], datapoints: list[Datapoint], distance: str = "eucl"):
    """
    Takes the query data given by the user and calculates its distances from the other datapoints. Distance metric
    is also chosen by the user, but the default value is Euclidean distance.
    Returns a dictionary whose key is the datapoint in the training data, and the value is the distance between that
    training datapoint and query datapoint.

    :param feature: the parsed and validated query point.
    :param datapoints: list of Datapoint objects.
    :param distance: the distance metric chosen by the user.
    :return: list of tuples where the first tuple element is the distance from query point,
     and second tuple element is the Datapoint itself.
    """
    if distance == "manh":
        metric = manhattan
    elif distance == "cos":
        metric = cosine
    else:
        metric = euclidean

    feature_distances = []

    for point in datapoints:
        pair = metric(feature, list(point.features)), point
        feature_distances.append(pair)

    return feature_distances

def k_nearest_points(k: int, distances: list[tuple[float, Datapoint]]) -> list[Datapoint]:
    """
    Finds the k smallest distances from the array of all distances.
    Returns a sub-array containing the smallest distance.

    :param k: number of neighbors with which the comparison has to be made.
    :param distances: an array containing the distances between the query data and the training data points
    :return: a sub-array pof distances which has the k smallest distances
    """
    sorted_distances = sorted(distances, key = lambda x: x[0])
    k_nearest = sorted_distances[:k]
    return [x[1] for x in k_nearest]

def get_classification(nearest_neighbors: list[Datapoint]):
    """
    Uses frequency-based majority voting to obtain a classification for the query data point.

    :param nearest_neighbors: the list of k nearest Datapoint objects.
    :return: the prediction of the query datapoint based on the KNN algorithm.
    """
    categories = [c.category for c in nearest_neighbors]
    counter = Counter(categories)
    return counter.most_common(1)[0][0]

