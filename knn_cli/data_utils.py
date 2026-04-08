import os.path
from dataclasses import dataclass
from enum import Enum
from re import split

import typer
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
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

ERR_COLOR = typer.colors.RED

def get_column_values(datapoints: list[Datapoint], feature_map: dict[str, int]) -> dict[str, list[float]]:
    """
    Extracts all values for each feature column across the dataset.

    :param datapoints: list of Datapoint objects representing the training example_datasets.
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
    """
    Prompts the user to enter a dataset file path and validates its existence.

    Continuously requests input until a valid file path is provided. A path is
    considered valid if it exists and points to a file on the filesystem.

    Displays a styled error message for invalid paths and re-prompts the user.

    :return: string representing a valid dataset file path.
    """
    while True:
        path = typer.prompt("Enter dataset path")
        if os.path.isfile(path):
            break
        typer.echo(typer.style(f"Dataset file: \"{path}\" is invalid.", fg=ERR_COLOR))

    return path

def get_valid_k(datapoints: list[Datapoint]) -> int:
    """
    Prompts the user to enter a valid value of k for the KNN algorithm.

    Continuously requests input until a valid integer k is provided. A valid k
    must be strictly greater than 0 and strictly less than the number of datapoints in
    the dataset.

    Displays a styled error message for invalid values and re-prompts the user.

    :param datapoints: list of Datapoint objects representing the dataset.
    :return: integer representing a valid k value within the range (0, len(datapoints)).
    """
    n = len(datapoints)
    while True:
        k_value = typer.prompt("Enter the value of k", type=int)
        if 0 < k_value < n:
            typer.echo(" ")
            break

        typer.echo(typer.style(f"The value of k ({k_value}) must be positive, and not exceed the size of the dataset ({n}).", fg=ERR_COLOR))

    return k_value

def display_column_names(console: Console, columns: list[str]) -> None:
    """
    Displays the available dataset column names in a formatted table.

    Organizes column names into a fixed number of columns per row and renders
    them using a rich table for improved readability. Adds spacing between rows
    for visual separation.

    :param console: Rich Console object used to render the table.
    :param columns: list of strings representing the dataset column names.

    :return: None
    """
    table = Table(title="Available Columns")

    cols_per_row = 4
    for _ in range(cols_per_row):
        table.add_column("")

    for i in range(0, len(columns), cols_per_row):
        chunk = columns[i:i + cols_per_row]
        chunk += [""] * (cols_per_row - len(chunk))
        table.add_row(*chunk)
        table.add_row(*[""] * cols_per_row)

    console.print(table)
    typer.echo(" ")

def get_valid_categorical_label(columns: list[str]):
    """
    Prompts the user to enter a valid categorical label (column name).

    Continuously requests input until the provided label matches one of the
    available dataset columns.

    Displays a styled error message for invalid column names and re-prompts
    the user.

    :param columns: list of strings representing the dataset column names.

    :return: string representing a valid categorical label.
    """
    temp_set = set(columns)
    while True:
        label = typer.prompt("Enter the column name of the categorical variable")
        if label in temp_set:
            typer.echo(" ")
            break
        typer.echo(typer.style(f"Column name \"{label}\" not found in dataset.", fg=ERR_COLOR))

    return label

def get_normalization_requirement():
    """
    Prompts the user to determine whether feature normalization should be applied,
    and if so, selects a valid normalization method.

    First asks the user to confirm whether normalization is required. If enabled,
    continuously requests input until a valid normalization method ("zscore" or
    "minmax") is provided.

    Displays a styled error message for invalid method inputs and re-prompts
    the user.

    :return: NormalizationMethods enum corresponding to the selected method,
    or None if normalization is not enabled.
    """
    normalize_method = None
    wants_normalization = typer.confirm("Enable feature normalization?")
    if wants_normalization:
        while True:
            normalize_method = typer.prompt("Enter normalization method (zscore/minmax)")
            if normalize_method.lower() in {"zscore", "minmax"}:
                typer.echo(" ")
                break
            typer.echo(typer.style("Normalization method must be either \"zscore\" or \"minmax\".", fg=ERR_COLOR))

    return NormalizationMethods(normalize_method.lower()) if normalize_method else None

def get_model_pathway():
    """
    Prompts the user to select a valid model usage pathway.

    Continuously requests input until the user provides either "classification"
    or "evaluation". Input is normalized by stripping whitespace and converting
    to lowercase before being returned.

    This also ensures that the user cannot invoke both pathways simultaneously.

    Displays a styled error message for invalid inputs and re-prompts the user.

    :return: string representing the selected model pathway ("classification" or "evaluation").
    """
    while True:
        pathway = typer.prompt("Enter model usage pathway (classification/evaluation)")
        if pathway.lower() in {"classification", "evaluation"}:
            typer.echo(" ")
            break

        typer.echo(typer.style("Model pathway must be either \"classification\" or \"evaluation\".", fg=ERR_COLOR))

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
            raise ValueError("Query datapoint contains non-numerical example_datasets.")

    return result

def get_query_input(feature_index_map):
    """
    Prompts the user to enter values for each feature in the dataset.

    Iterates through the feature index map and requests a numeric input for
    each feature. Inputs are collected in order and stored as floating-point
    values.

    :param feature_index_map: dictionary mapping each feature name to its
        corresponding index.

    :return: list of floats representing the query point in feature order.
    """
    values = []
    for feature in feature_index_map:
        val = typer.prompt(f"Enter feature value for \"{feature}\"", type = float)
        values.append(val)
        typer.echo(" ")

    return values

def get_valid_dist_metric():
    """
    Prompts the user to enter a valid distance metric for the KNN algorithm.

    Continuously requests input until a valid metric identifier ("eucl",
    "manh", or "cos") is provided.

    Displays a styled error message for invalid inputs and re-prompts
    the user.

    :return: Distances enum corresponding to the selected metric.
    """
    metrics = {"eucl", "manh", "cos"}
    while True:
        metric = typer.prompt("Enter distance metric for algorithm (eucl/manh/cos)")
        if metric.lower() in metrics:
            typer.echo(" ")
            break
        typer.echo(typer.style("Distance metric must be either \"eucl\", \"manh\", or \"cos\"", fg=ERR_COLOR))

    return Distances(metric.lower())

def get_valid_plot_args(feature_index_map):
    """
    Prompts the user to configure plotting options and validates selected feature columns.

    First asks whether plotting should be enabled. If enabled, requests valid
    feature column names for the x-axis and y-axis. Optionally supports 3D plotting
    by prompting for a z-axis feature.

    Performs case-insensitive validation of feature names while preserving original
    column naming for output consistency.

    Displays styled error messages for invalid feature inputs and re-prompts
    the user.

    :param feature_index_map: dictionary mapping each feature name to its
        corresponding index.

    :return: tuple containing:
        - boolean indicating whether plotting is enabled
        - string representing x-axis feature (or None)
        - string representing y-axis feature (or None)
        - string representing z-axis feature (or None)
    """
    original = {feature.lower() : feature for feature in feature_index_map}
    plot = typer.confirm("Enable plotting?")
    x, y, z = None, None, None
    if plot:
        while True:
            x = typer.prompt("Enter feature column for x-axis")
            if x.lower() in original:
                typer.echo(" ")
                break
            typer.echo(typer.style(f"Feature column \"{x}\" does not exist in dataset.", fg=ERR_COLOR))

        while True:
            y = typer.prompt("Enter feature column for y-axis")
            if y.lower() in original:
                typer.echo(" ")
                break
            typer.echo(typer.style(f"Feature column \"{y}\" does not exist in dataset.", fg=ERR_COLOR))

        wants_3d_plot = typer.confirm("Enable 3D plotting?")
        if wants_3d_plot:
            while True:
                z = typer.prompt("Enter feature column for z-axis")
                if z.lower() in original:
                    typer.echo(" ")
                    break
                typer.echo(typer.style(f"Feature column \"{z}\" does not exist in dataset.", fg=ERR_COLOR))

    if not plot:
        return plot, None, None, None
    else:
        return plot, original[x.lower()], original[y.lower()], original[z.lower()] if z is not None else None

def get_valid_tts_fraction():
    """
    Prompts the user to enter a valid train-test split fraction.

    Continuously requests input until a floating-point value strictly between
    0 and 1 is provided.

    Displays a styled error message for invalid values and re-prompts
    the user.

    :return: float representing a valid train-test split fraction in the interval (0, 1).
    """
    while True:
        frac = typer.prompt("Enter train-test-split fraction", type=float)
        if 0 < frac < 1:
            typer.echo(" ")
            break
        typer.echo(typer.style("Train test split fraction must lie in the open interval (0, 1)", fg=ERR_COLOR))

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

def landing_message(console: Console):
    """
    Displays the landing message for the CLI with usage guidance.

    Renders a styled panel using Rich that provides an overview of required
    inputs, supported options, and expected formats for interacting with the tool.
    Covers dataset requirements, parameter constraints, and available configuration
    choices.

    :param console: Rich Console object used to render the panel.

    :return: None
    """
    console.print(Panel(
        Align.center(
            "[bold cyan]KNN Command Line Tool[/bold cyan]\n"
            "[dim]Real World Data Analysis[/dim]\n\n"
            "[bold white]Dataset Path[/bold white]\n"
            "[dim]Path to a CSV file with a header row and at least 2 columns[/dim]\n\n"
            "[bold white]k[/bold white]\n"
            "[dim]Number of nearest neighbors to consider. Must be positive and less than dataset size[/dim]\n\n"
            "[bold white]Distance Metric[/bold white]\n"
            "[dim]eucl - Euclidean  |  manh - Manhattan  |  cos - Cosine[/dim]\n\n"
            "[bold white]Normalization[/bold white]\n"
            "[dim]zscore - Z-score standardization  |  minmax - Min-max scaling[/dim]\n\n"
            "[bold white]Pathway[/bold white]\n"
            "[dim]classification - Predict a query point  |  evaluation - Measure model accuracy[/dim]\n\n"
            "[bold white]Train-Test Split[/bold white]\n"
            "[dim]Fraction of example_datasets used for testing e.g. 0.2 = 80/20 split[/dim]\n\n"
            "[dim]Press Ctrl+C to exit at any time[/dim]"
        ),
        border_style="cyan",
        padding=(1, 4)
    ))




