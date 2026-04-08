from knn_cli.data_utils import Datapoint
from knn_cli.distance_metric import euclidean, manhattan, cosine
from collections import Counter

def calculate_distances(feature: list[float], datapoints: list[Datapoint],
                        distance: str = "eucl") -> list[tuple[float, Datapoint]]:
    """
    Calculates the distance between the query point and every datapoint in the
    training set using the specified distance metric.

    Defaults to Euclidean distance if an unrecognized metric string is provided.

    :param feature: the parsed query point as a list of floats.
    :param datapoints: list of Datapoint objects representing the training example_datasets.
    :param distance: the distance metric to use. Accepts 'eucl', 'manh', or 'cos'.
    Defaults to 'eucl'.

    :return: list of tuples where each tuple contains the distance from the query
    point as a float and the corresponding Datapoint object.
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
    Selects the k closest datapoints from the precomputed distances list.

    :param k: number of nearest neighbors to select.
    :param distances: list of tuples where each tuple contains a distance float
    and its corresponding Datapoint object.

    :return: list of the k nearest Datapoint objects, ordered closest first.
    """
    sorted_distances = sorted(distances, key = lambda x: x[0])
    k_nearest = sorted_distances[:k]
    return [x[1] for x in k_nearest]

def get_classification(nearest_neighbors: list[Datapoint]) -> str:
    """
    Determines the predicted class of the query point using frequency-based
    majority voting across the k nearest neighbors.

    In the case of a tie, the first encountered class in the neighbors list wins,
    making the result deterministic.

    :param nearest_neighbors: list of the k nearest Datapoint objects.
    :return: the predicted category label as a string.
    """
    categories = [c.category for c in nearest_neighbors]
    counter = Counter(categories)
    return counter.most_common(1)[0][0]

