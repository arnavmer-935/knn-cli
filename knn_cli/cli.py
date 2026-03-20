from dataclasses import dataclass

import typer
from typing import Annotated

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from knn_cli.data_loader import load_dataset
from knn_cli.knn import calculate_distances, get_classification, k_nearest_points
from knn_cli.visualization import generate_plots

from knn_cli.data_utils import (Distances,
                                get_column_values,
                                validate_prediction_args,
                                validate_dataset_args,
                                get_valid_query_point, NormalizationMethods, Datapoint, KNNConfig, NORM_LABEL,
                                DISTANCE_LABEL, DescriptiveStats
                                )

from knn_cli.statistics import (mean_dataset,
                                median_dataset,
                                count_min_max,
                                quartile_values_dataset,
                                standard_deviation_dataset,
                                generate_desc_statistics
                                )

from knn_cli.normalization import get_mean_std_map, get_min_max_map, \
    get_normalized_datapoints, normalize_query_point_minmax, normalize_dataset_zscore, normalize_query_point_zscore, \
    normalize_dataset_minmax


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

def display_config(config: KNNConfig) -> None:
    """
    Displays a formatted table summarizing the user's configuration before execution.
    Called only when the --confirm flag is set, giving the user a chance to review
    all arguments prior to running the classification algorithm.

    :param config: #TODO

    :return: None
    """
    cli_console = Console()
    config_table = Table()
    config_table.add_column("Attribute")
    config_table.add_column("Value")

    x = "N/A" if config.x is None else config.x
    y = "N/A" if config.y is None else config.y
    z = "N/A" if config.z is None else config.z
    nm = "N/A" if config.normalize is None else NORM_LABEL[config.normalize]
    query = "N/A" if config.query_pt is None else config.query_pt

    tts_msg = None
    if config.tts is None:
        tts_msg = "N/A"

    else:
        training_percent = "N/A" if config.tts is None else f"{(1 - config.tts) * 100:.2f}%"
        testing_percent = "N/A" if config.tts is None else f"{config.tts * 100:.2f}%"
        tts_msg = f"{training_percent} training, {testing_percent} testing"

    config_table.add_row("Dataset", config.dataset)
    config_table.add_row("k", str(config.k))
    config_table.add_row("Query Datapoint", str(query))
    config_table.add_row("Distance Metric", DISTANCE_LABEL[config.distance])
    config_table.add_row("Train-Test split", tts_msg)
    config_table.add_row("Normalization Method", nm)
    config_table.add_row("Enable Descriptive Statistics?", str(config.describe))
    config_table.add_row("Enable Plotting?", str(config.plot))
    config_table.add_row("Plot Label for x-axis", x)
    config_table.add_row("Plot Label for y-axis", y)
    config_table.add_row("Plot Label for z-axis", z)

    cli_console.print(config_table, justify="center")

def main(
    dataset: str = typer.Argument("-d", help="a comma separated training data file."),
    k: int = typer.Argument("-k", help="the number of nearest neighbors to be considered."),
    query_data: Annotated[str, typer.Option("--p", help="the query data point's feature values separated by whitespace.")] = None,
    distance: Annotated[Distances, typer.Option("--m", help = "the distance metric", case_sensitive = False)] = Distances.eucl,
    tts: Annotated[float, typer.Option("--train-test-split", help="splits dataset into train and test sets based on the given fraction. "
                                                                  "e.g. 0.2 uses an 80/20 split for training/testing respectively.")] = None,
    normalize: Annotated[NormalizationMethods, typer.Option("--normalize", help="normalizes feature values using z-scores/min-max scaling.")] = None,
    describe: Annotated[bool, typer.Option(help="display in the standard output descriptive statistics of the data")] = False,
    plot: Annotated[bool, typer.Option(help="enables plotting mode")] = False,
    x: Annotated[str, typer.Option(help="label for the x axis")] = None,
    y: Annotated[str, typer.Option(help="label for the y axis")] = None,
    z: Annotated[str, typer.Option(help="label for the z axis (used only in 3D plots")] = None,
    confirm: Annotated[bool, typer.Option("--confirm", help="confirm arguments before running")] = False
) -> None:

    """
    The main entry point of the CLI. Responsible for parsing command line arguments,
    validating inputs, classifying the query point using the KNN algorithm, and
    optionally displaying descriptive statistics and scatter plots.

    Catches and reports ValueError for all invalid argument combinations, including
    invalid file paths, out-of-range k values, query point dimension mismatches,
    and malformed plot configurations. Exits with code 1 on any validation failure.

    :param dataset: file path of the training dataset.
    :param k: number of nearest neighbors to be considered.
    :param query_data: whitespace-separated feature values of the query point.
    :param distance: the distance metric to use. Accepts eucl, manh, or cos. Defaults to eucl.
    :param tts: the fraction required for train-test splitting and evaluating model accuracy.
    :param describe: if True, displays a descriptive statistics table for the dataset.
    :param plot: if True, enables scatter plot generation.
    :param x: feature name to plot on the x-axis. Requires --plot.
    :param y: feature name to plot on the y-axis. Requires --x.
    :param z: feature name to plot on the z-axis. Produces a 3D plot. Requires --y.
    :param confirm: if True, displays a configuration summary and prompts the user to proceed.
    :param normalize: the normalization method to be used for scaling the data values.
    must be either "zscore" or "minmax".

    :return: None
    """

    console = Console()
    try:
        validate_prediction_args(dataset, k, normalize, tts)
        query_point = get_valid_query_point(query_data) if query_data is not None else None
        user_datapoints, feature_map = load_dataset(dataset)
        validate_dataset_args(user_datapoints, feature_map, k, query_data, plot, x, y, z, tts)

        categories = sorted({pt.category for pt in user_datapoints})

        user_config = KNNConfig(dataset, k, query_point, categories, distance, normalize, describe, plot, x, y, z, tts)

        if confirm:
            console.rule("[bold cyan] Configuration Attributes")
            display_config(user_config)

            if not typer.confirm("Proceed?"):
                raise typer.Exit()

        column_values = get_column_values(user_datapoints, feature_map)
        mean_of_data = mean_dataset(column_values)
        st_devs = standard_deviation_dataset(column_values)
        data_count, data_min, data_max = count_min_max(column_values)

        normalized_values = None
        normalized_pt = None

        if normalize == NormalizationMethods.zscore:
            mean_std_map = get_mean_std_map(mean_of_data, st_devs)
            normalized_values = normalize_dataset_zscore(user_datapoints, feature_map, mean_std_map)
            normalized_pt = normalize_query_point_zscore(feature_map, mean_std_map, query_point)

        elif normalize == NormalizationMethods.minmax:
            min_max_map = get_min_max_map(data_min, data_max)
            normalized_values = normalize_dataset_minmax(user_datapoints, feature_map, min_max_map)
            normalized_pt = normalize_query_point_minmax(feature_map, min_max_map, query_point)

        normalized_user_points = None
        if normalized_values is not None:
            normalized_user_points = get_normalized_datapoints(user_datapoints, normalized_values, feature_map)

        user_computation_config = Computation(
            user_datapoints,
            query_point,
            column_values,
            feature_map,
            normalized_user_points,
            normalized_pt,
            mean_of_data,
            st_devs,
            data_count,
            data_min,
            data_max
        )

        if tts is None:
            classification_and_analysis(console, user_config, user_computation_config)

        else:
            evaluation()

    except ValueError as e:
        print(e)
        raise typer.Exit(code=1)


def classification_and_analysis(console: Console, knn_config: KNNConfig, computation: Computation):
    """
    should contain plot generation, descriptive stats, classification logic
    :return: None
    """

    requirements = computation.normalized_query, computation.normalized_datapoints
    query_data_prediction = None

    if any(x is None for x in requirements):
        distances = calculate_distances(knn_config.query_pt, computation.datapoints, knn_config.distance)
        k_nearest_dists = k_nearest_points(knn_config.k, distances)
        query_data_prediction = get_classification(k_nearest_dists)

    else:
        normalized_distances = calculate_distances(requirements[0], requirements[1], knn_config.distance)
        k_nearest_dists = k_nearest_points(knn_config.k, normalized_distances)
        query_data_prediction = get_classification(k_nearest_dists)

    console.print(
        Panel(
            Align.center(f"[bold green]{query_data_prediction}[/bold green]"),
            title="Prediction",
            border_style="green"
        )
    )

    console.rule("[bold cyan]Dataset Summary")
    print("Number of features:", len(computation.feature_map))
    categories = knn_config.categories
    cat_str = ""
    for i in range(len(categories)):
        ct = categories[i][0].upper() + categories[i][1:].lower()

        if i == len(categories) - 1:
            cat_str += ct
        else:
            cat_str += ct + ", "

    print("Categories:", cat_str)
    print()

    if knn_config.describe:
        console.rule("[bold cyan]Descriptive Statistics")
        median_of_data = median_dataset(computation.column_values)
        q1, q3 = quartile_values_dataset(computation.column_values)

        stats_config = DescriptiveStats(
            computation.mean_of_data,
            computation.count,
            computation.min_dict,
            computation.max_dict,
            median_of_data,
            q1, q3,
            computation.st_devs
        )

        generate_desc_statistics(stats_config)

    if knn_config.plot:
        generate_plots(computation.datapoints, computation.feature_map, knn_config.k, knn_config.query_pt,
                       knn_config.x, knn_config.y, knn_config.z)

def evaluation():
    """
    should contain train, test splitting, eval metrics. Cannot be invoked simultaneously with
    classification_and_analysis.
    :return: None
    """
    pass



if __name__ == '__main__':
    typer.run(main)