from .data_utils import find_min, Datapoint
from .distance_metric import euclidean, manhattan, cosine

def calculate_distances(feature: list[float], datapoints: list[Datapoint], distance: str = "eucl"): #feature = query datapoint
    """
    Takes the query data given by the user and calculates its distances from the other datapoints. Distance calculation
    method is also chosen by the user, but the default value is Euclidean distance.
    Returns a dictionary whose key is the datapoint in the training data, and the value is the distance between that
    training datapoint and query datapoint.
    :param feature: query datapoint entered by user
    :param datapoints: list of Datapoint Objects
    :param distance: the distance metric chosen by the user
    :return: dictionary whose key is the datapoint in the training data, and the value is the distance between that
    training datapoint and query datapoint.
    """
    feature_distance_map = {}
    if distance == "manh":
        for point in datapoints:
            feature_distance_map[point.features] = manhattan(feature, list(point.features))
    elif distance == "cos":
        for point in datapoints:
            feature_distance_map[point.features] = cosine(feature, list(point.features))
    else:
        for point in datapoints:
            feature_distance_map[point.features] = euclidean(feature, list(point.features))
    return [feature_distance_map, list(feature_distance_map.values())]
    #i = 0: dict key is tuple(feature values), value is distance from user's entered point
    #i = 1: list containing all distances from user's feature point


def k_nearest_distances(k: int, distances: list[float]) -> list[float]:
    """
    Finds the k smallest distances from the array of all distances. The value of k is given by the user.
    Returns a sub-array containing the smallest distance.
    :param k: number of neighbors with which comparison has to be made
    :param distances: an array containing the distances between the query data and the training data points
    :return: a sub-array pof distances which has the k smallest distances
    """
    nearest_distances = []
    while len(nearest_distances) < k:
        min_distance = distances.pop(find_min(distances))
        nearest_distances.append(min_distance)
    return nearest_distances

def get_classification(nearest_distances: list[float], feature_distance_map, dataset_points: list[Datapoint]):

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
    k_nearest_datapoints = [] #contains feature points of the k nearest neighbors
    k_nearest_neighbors = [] #contains the categories of the k nearest neighbors
    for feature in feature_distance_map:
        if feature_distance_map[feature] in nearest_distances:
            k_nearest_datapoints.append(feature)
        else:
            continue

    for pt in k_nearest_datapoints:
        for point in dataset_points:
            if point.features == pt:
                k_nearest_neighbors.append(point.category)
            else:
                continue

    knn_hashtable = {}
    for category in k_nearest_neighbors:
        if category not in knn_hashtable:
            knn_hashtable[category] = 1
        else:
            knn_hashtable[category] += 1

    classification = k_nearest_neighbors[0] #chooses the first category which appears in the knn list
    for c in knn_hashtable:
        if knn_hashtable[c] > knn_hashtable[classification]:
            classification = c
        else:
            continue
    return classification