import os.path
from dataclasses import dataclass
from enum import Enum
from re import split

@dataclass
class Datapoint:
    features: tuple[float,...]
    category: str

class Distances(str, Enum):
    eucl = "eucl"
    manh = "manh"
    cos = "cos"

class NormalizationMethods(str, Enum):
    zscore = "zscore"
    minmax = "minmax"

def get_column_values(datapoints: list[Datapoint], feature_map: dict[str, int]) -> dict[str, list[float]]:
    """
    Extracts all values for each feature column across the dataset.

    :param datapoints: list of Datapoint objects representing the training data.
    :param feature_map: dictionary mapping each feature column name to its 0-based index.

    :return: dictionary mapping each feature column name to a list of all its values
    across the dataset.
    """
    column_map = {feature : [] for feature in feature_map}

    for point in datapoints:
        for feature, idx in feature_map.items():
            column_map[feature].append(point.features[idx])

    return column_map

def median(arr: list[float]) -> float:
    """
    Calculates the median of a numerical array.
    Returns 0 if the array is empty.

    :param arr: list of numerical values.

    :return: median value of the array, or 0 if the array is empty.
    """
    arr = sorted(arr)
    n = len(arr)
    if not arr:
        return 0
    elif n % 2 == 0:
        return 0.5 * (arr[n//2 - 1] + arr[n//2])
    else:
        return arr[n//2]

def get_categories(dataset: str) -> list[str]:
    """
    Extracts the unique category values present in the dataset.

    :param dataset: file path of the training dataset.

    :return: list of unique category values found in the last column of the dataset.
    """
    with open(dataset) as f:
        ls = list(set([l.strip().split(",")[-1] for l in f.readlines()]))
        return ls

def validate_prediction_args(dataset: str, k: int, normalize: str) -> None:
    """
    Performs initial validation on the dataset path, the k value, and the normalization method
    before the dataset is loaded.

    Raises ValueError if the dataset path does not point to an existing file, if k is zero or negative,
    or if the normalization method is not "zscore" or "minmax"

    :param dataset: file path of the training dataset.
    :param k: number of nearest neighbors to be considered.
    :param normalize: the method for feature normalization.

    :return: None
    """
    if not os.path.isfile(dataset):
        raise ValueError(f"Dataset file: {dataset} is invalid.")

    if k <= 0:
        raise ValueError("The value of k must be positive.")

    feature_norm = normalize.strip().lower()

    if not feature_norm or feature_norm not in {"zscore", "minmax"}:
        raise ValueError("Invalid normalization method. Expected 'zscore' or 'minmax'.")

def validate_dataset_args(datapoints: list[Datapoint], feature_map: dict[str, int],
                          k: int, query_data: str, plot: bool, x: str, y: str, z: str) -> None:
    """
    Performs validation on the dataset and argument configuration after the dataset is loaded.

    Raises ValueError in the following cases:
    - k exceeds the number of datapoints in the dataset
    - the feature map is None or empty
    - the dataset contains fewer than 3 columns
    - the query point dimensions do not match the number of feature columns
    - axis arguments are provided without the --plot flag
    - z-axis is specified without a y-axis
    - y-axis is specified without an x-axis
    - any axis feature name does not exist in the dataset

    :param datapoints: list of Datapoint objects representing the training data.
    :param feature_map: dictionary mapping each feature column name to its 0-based index.
    :param k: number of nearest neighbors to be considered.
    :param query_data: whitespace-separated feature values of the query point as a raw string.
    :param plot: boolean flag indicating whether plotting mode is enabled.
    :param x: feature name assigned to the x-axis, or None if not specified.
    :param y: feature name assigned to the y-axis, or None if not specified.
    :param z: feature name assigned to the z-axis, or None if not specified.
    :return: None
    """
    if k > len(datapoints):
        raise ValueError(f"Value of k ({k}) cannot exceed the size of dataset ({len(datapoints)}).")

    if feature_map is None:
        raise ValueError("Dataset is empty or malformed.")

    if len(feature_map) < 2:
        raise ValueError("Insufficient columns in dataset file. Expected at least 2 columns.")

    if len(get_valid_query_point(query_data)) != len(feature_map):
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

def get_valid_query_point(query_point: str) -> list[float]:
    """
    Parses and validates a whitespace-separated string of feature values into a list of floats.
    Raises ValueError if the string is empty or contains any non-numeric values.

    :param query_point: whitespace-separated string of numeric feature values.

    :return: list of parsed float values representing the query point's features.
    """
    values = split(r'\s+', query_point.strip())
    result = []
    if not values:
        raise ValueError("Query datapoint is empty.")

    for val in values:
        try:
            numeric = float(val)
            result.append(numeric)
        except ValueError:
            raise ValueError("Query datapoint contains non-numerical data.")

    return result
