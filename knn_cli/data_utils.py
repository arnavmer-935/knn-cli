import os.path
from dataclasses import dataclass
from enum import Enum
from re import split

import typer
from rich.console import Console
from rich.table import Table

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

@dataclass
class DescriptiveStats:
    mean_of_data: dict
    count: float
    min_of_data: dict
    max_of_data: dict
    median_of_data: dict
    Q1_of_data: dict
    Q3_of_data: dict
    stdev_of_data: dict

DISTANCE_LABEL = {
    Distances.eucl: "Euclidean",
    Distances.manh: "Manhattan",
    Distances.cos: "Cosine Similarity"
}

NORM_LABEL = {
    NormalizationMethods.zscore: "Z-Score Normalization",
    NormalizationMethods.minmax: "Min-Max Scaling Normalization"
}

@dataclass
class KNNConfig:
    dataset: str
    k: int
    query_pt: list[float]
    categories: list[str]
    distance: Distances
    normalize: NormalizationMethods
    describe: bool
    plot: bool
    x: str
    y: str
    z: str
    tts: float

@dataclass
class Computation:
    datapoints: list[Datapoint]
    query_pt: list[float]
    column_values: dict[str, list[float]]
    feature_map: dict[str, int]
    normalized_datapoints: list[Datapoint]
    normalized_query: list[float]
    mean_of_data: dict
    st_devs: dict
    count: float
    min_dict: dict
    max_dict: dict

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

def get_valid_dataset_path() -> str:
    while True:
        path = typer.prompt("Enter dataset path")
        if os.path.isfile(path):
            break
        typer.echo(f"Dataset file: \"{path}\" is invalid.")

    return path

def get_valid_k(datapoints: list[Datapoint]) -> int:
    n = len(datapoints)
    while True:
        k_value = typer.prompt("Enter the value of k", type=int)
        if 0 < k_value < n:
            break
        typer.echo(f"The value of k ({k_value}) must be positive, and not exceed the size of the dataset ({n}).")

    return k_value


def display_column_names(console: Console, columns: list[str]) -> None:
    table = Table(title="Available Columns")

    cols_per_row = 4
    for _ in range(cols_per_row):
        table.add_column("")

    for i in range(0, len(columns), cols_per_row):
        chunk = columns[i:i + cols_per_row]
        chunk += [""] * (cols_per_row - len(chunk))
        table.add_row(*chunk)
        table.add_row(*[""] * cols_per_row)  # empty spacer row

    console.print(table)

def get_valid_categorical_label(columns: list[str]):
    temp_set = set(columns)
    while True:
        label = typer.prompt("Enter the column name of the categorical variable")
        if label in temp_set:
            break
        typer.echo(f"Column name {label} not found in dataset.")

    return label

def get_normalization_requirement():
    normalize_method = None
    wants_normalization = typer.confirm("Enable feature normalization?")
    if wants_normalization:
        while True:
            normalize_method = typer.prompt("Enter normalization method (zscore/minmax)")
            if normalize_method.lower() in {"zscore", "minmax"}:
                break
            typer.echo("Normalization method must be either \"zscore\" or \"minmax\".")

    return NormalizationMethods(normalize_method.lower()) if normalize_method else None

def get_model_pathway():
    while True:
        pathway = typer.prompt("Enter model usage pathway (classification/evaluation)")
        if pathway.lower() in {"classification", "evaluation"}:
            break
        typer.echo("Model pathway must be either \"classification\" or \"evaluation\".")

    return pathway.strip().lower()

def get_valid_query_point(query_point: str) -> list[float]:
    """
    Parses and validates a whitespace-separated string of feature values into a list of floats.
    Raises ValueError if the string is empty or contains any non-numeric values.

    :param query_point: whitespace-separated string of numeric feature values.

    :return: list of parsed float values representing the query point's features.
    """
    if query_point is None or not query_point:
        raise ValueError("Query point is not defined.")

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

def get_query_input(feature_index_map):
    values = []
    for feature in feature_index_map:
        val = typer.prompt(f"Enter feature value for \"{feature}\"", type = float)
        values.append(val)

    return values

def get_valid_dist_metric():
    metrics = {"eucl", "manh", "cos"}

    while True:
        metric = typer.prompt("Enter distance metric for algorithm (eucl/manh/cos)")
        if metric in metrics:
            break
        typer.echo("Distance metric must be either \"eucl\", \"manh\", or \"cos\"")

    return Distances(metric.lower())

def get_valid_plot_args(feature_index_map):
    original = {feature.lower() : feature for feature in feature_index_map}
    plot = typer.confirm("Enable plotting?")
    x, y, z = None, None, None
    if plot:
        while True:
            x = typer.prompt("Enter feature column for x-axis")
            if x.lower() in original:
                break
            typer.echo(f"Feature column \"{x}\" does not exist in dataset.")

        while True:
            y = typer.prompt("Enter feature column for y-axis")
            if y.lower() in original:
                break
            typer.echo(f"Feature column \"{y}\" does not exist in dataset.")

        wants_3d_plot = typer.confirm("Enable 3D plotting?")
        if wants_3d_plot:
            while True:
                z = typer.prompt("Enter feature column for z-axis")
                if z.lower() in original:
                    break
                typer.echo(f"Feature column \"{z}\" does not exist in dataset.")

    if not plot:
        return plot, None, None, None
    else:
        return plot, original[x.lower()], original[y.lower()], original[z.lower()] if z is not None else None

def get_valid_tts_fraction():
    while True:
        frac = typer.prompt("Enter train-test-split fraction", type=float)
        if 0 < frac < 1:
            break
        typer.echo("Train test split fraction must lie in the open interval (0, 1)")

    return frac

def get_format_color(improvement: float) -> str:
    """
    Returns a Rich-compatible color string based on the magnitude of the
    accuracy improvement over the baseline.

    :param improvement: the difference between model accuracy and baseline accuracy.

    :return: a color string — "red" for negative, "yellow" for negligible,
    "cyan" for modest, and "green" for strong improvement.
    """
    if improvement < 0:
        return "red"

    elif improvement < 0.02:
        return "yellow"

    elif improvement < 0.08:
        return "cyan"

    else:
        return "green"

def get_improvement_interpretation(improvement: float) -> str:
    """
    Returns a human-readable interpretation of the model's accuracy improvement
    over the baseline for display in the evaluation output panel.

    :param improvement: the difference between model accuracy and baseline accuracy.

    :return: a string describing the significance of the improvement.
    """
    if improvement < 0:
        return "Model accuracy is worse than baseline accuracy."
    elif improvement < 0.02:
        return "Model has produced no meaningful improvement."
    elif improvement < 0.08:
        return "Model has produced a modest improvement."
    elif improvement < 0.2:
        return "Model has produced a strong improvement."
    else:
        return "Model has produced a very strong improvement (please verify dataset)."





