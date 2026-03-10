import os.path
from dataclasses import dataclass
from enum import Enum

@dataclass
class Datapoint:
    features: tuple[float,...]
    category: str

class Distances(str, Enum):
    eucl = "Euclidean"
    manh = "Manhattan"
    cos = "Cosine Similarity"

def get_column_values(datapoints, feature_map):
    """
    Helper method for getting all the values of each column.
    :param dataset: file path for data
    :return: a dictionary whose key is the column name and value is a list containing all of its column values.
    """
    column_map = {feature : [] for feature in feature_map}

    for point in datapoints:
        for feature, idx in feature_map.items():
            column_map[feature].append(point.features[idx])

    return column_map #returns dict with key as i/d variable name and value as all its data in a list

def median(arr):
    """
    Helper method to find the median of a numerical array.
    :param arr: numerical array
    :return: median of that array
    """
    arr = sorted(arr)
    n = len(arr)
    if not arr:
        return 0
    elif n % 2 == 0:
        return 0.5 * (arr[n//2 - 1] + arr[n//2])
    else:
        return arr[n//2]

def get_categories(dataset):
    """
    Helper method for getting the different category values in the data
    :param dataset: file path for data
    :return: list of categories
    """
    with open(dataset) as f:
        ls = list(set([l.strip().split(",")[-1] for l in f.readlines()]))
        return ls


def validate_prediction_args(dataset: str, k: int, query_data):
    if not os.path.isfile(dataset):
        raise ValueError(f"Dataset file: {dataset} is invalid.")

    if k <= 0:
        raise ValueError("The value of k must be positive.")

    query_pt = query_data.strip().split(" ")
    if not query_pt:
        raise ValueError("Query datapoint is empty.")

    for val in query_pt:
        try:
            numeric = float(val)
        except ValueError:
            raise ValueError("Query datapoint contains non-numerical data.")

def validate_dataset_args(datapoints, feature_map, k, query_data, plot, x, y, z):
    if k > len(datapoints):
        raise ValueError(f"Value of k ({k}) cannot exceed the size of dataset ({len(datapoints)}).")

    if feature_map is None:
        raise ValueError("Dataset is empty or malformed.")

    if len(feature_map) <= 2:
        raise ValueError("Insufficient columns in dataset file. Expected at least 3 columns.")

    query_pt = query_data.strip().split()
    if len(query_pt) != len(feature_map):
        raise ValueError("Number of feature columns in dataset does not match the dimensions of query point.")

    if not plot and any(axis is not None for axis in (x, y, z)):
        raise ValueError("Axis arguments (--x, --y, --z) require the --plot flag.")

    if z is not None and y is None:
        raise ValueError("z-axis requires a y-axis feature.")

    if y is not None and x is None:
        raise ValueError("y-axis requires an x-axis feature.")

    for axis, feature in {"x": x, "y": y, "z": z}.items():
        if feature is not None and feature not in feature_map:
            raise ValueError(f"{axis}-axis feature \"{feature}\" does not exist in dataset.")