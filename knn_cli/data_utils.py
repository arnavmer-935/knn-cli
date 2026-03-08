import os.path
import csv
from dataclasses import dataclass
from enum import Enum

@dataclass
class Datapoint:
    features: tuple[float,...]
    category: str

class Distances(str, Enum):
    eucl = "eucl"
    manh = "manh"
    cos = "cos"

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

    query_pt = query_data.strip().split()
    if len(query_pt) == 0:
        raise ValueError("Query datapoint is empty.")

    for val in query_pt:
        try:
            numeric = float(val)
        except ValueError:
            raise ValueError("Query datapoint contains non-numerical data.")

def validate_dataset_args(datapoints, feature_map, k, query_pt, x, y, z):
    if k > len(datapoints):
        raise ValueError("Value of k cannot exceed the size of dataset.")

    if feature_map is None:
        raise ValueError("Dataset is empty or malformed.")

    if len(feature_map) <= 2:
        raise ValueError("Insufficient columns in dataset file. Expected at least 3 columns.")

    if len(query_pt) != len(feature_map):
        raise ValueError("Number of feature columns in dataset does not match the dimensions of query point.")

    if x is not None:
        if x not in feature_map:
            raise ValueError(f"Feature column {x} does not exist in dataset.")

    if y is not None:
        if y not in feature_map:
            raise ValueError(f"Feature column {y} does not exist in dataset.")

    if z is not None:
        if z not in feature_map:
            raise ValueError(f"Feature column {z} does not exist in dataset.")