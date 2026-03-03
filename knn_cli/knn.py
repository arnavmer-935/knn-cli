from .data_utils import Datapoint
from .distance_metric import euclidean, manhattan, cosine
from collections import Counter

#TODO: Rewrite Documentation

def calculate_distances(feature: list[float], datapoints: list[Datapoint], distance: str = "eucl"): #feature = query datapoint
    """
    Takes the query data given by the user and calculates its distances from the other datapoints. Distance calculation
    method is also chosen by the user, but the default value is Euclidean distance.
    Returns a dictionary whose key is the datapoint in the training data, and the value is the distance between that
    training datapoint and query datapoint.
    :param feature: query datapoint entered by user
    :param datapoints: list of Datapoint Objects
    :param distance: the distance metric chosen by the user
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
    Finds the k smallest distances from the array of all distances. The value of k is given by the user.
    Returns a sub-array containing the smallest distance.
    :param k: number of neighbors with which comparison has to be made
    :param distances: an array containing the distances between the query data and the training data points
    :return: a sub-array pof distances which has the k smallest distances
    """
    if k <= 0:
        raise ValueError("The value of k must be positive")

    sorted_distances = sorted(distances, key = lambda x: x[0]) #sort by distance from datapoint
    k_nearest = sorted_distances[:k]
    return [x[1] for x in k_nearest]

def get_classification(nearest_neighbors: list[Datapoint]):

    """
    Uses the k nearest distances, a dictionary whose key is a training datapoint and value is distance from the point,
    and a list of Datapoint objects to find the categories of the nearest neighbors.
    Returns an array containing the k nearest neighbors' categories
    :param nearest_distances: the k smallest distances between the query datapoint and training data datapoint.
    :param feature_distance_map: a dictionary whose key is a datapoint in the training data, and value is the distance
    between that point and query datapoint.
    :param dataset_points: the list of datapoint objects obtained from loading the data
    :return: the prediction of the query datapoint based on the KNN algorithm
    """

    categories = [c.category for c in nearest_neighbors]
    counter = Counter(categories)
    return counter.most_common(1)[0][0]

