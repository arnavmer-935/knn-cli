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
                                get_valid_query_point, NormalizationMethods
                                )

from knn_cli.statistics import (mean_dataset,
                                median_dataset,
                                count_min_max,
                                quartile_values_dataset,
                                standard_deviation_dataset,
                                generate_desc_statistics
                                )

from knn_cli.normalization import get_mean_std_map, normalized_values_zscore, get_min_max_map, normalized_values_minmax, \
    get_normalized_datapoints

DISTANCE_LABEL = {
    Distances.eucl: "Euclidean",
    Distances.manh: "Manhattan",
    Distances.cos: "Cosine Similarity"
}

NORM_LABEL = {
    NormalizationMethods.zscore: "Z-Score Normalization",
    NormalizationMethods.minmax: "Min-Max Scaling Normalization"
}

def display_config(dataset: str, k: int, query_pt: list[float], distance: Distances, normalize: NormalizationMethods,
                   describe: bool, plot: bool, x: str, y: str, z: str) -> None:
    """
    Displays a formatted table summarizing the user's configuration before execution.
    Called only when the --confirm flag is set, giving the user a chance to review
    all arguments prior to running the classification algorithm.

    :param dataset: file path of the training dataset.
    :param k: number of nearest neighbors to be considered.
    :param query_pt: the parsed query point as a list of floats.
    :param distance: the selected Distances enum member representing the distance metric.
    :param normalize: #TODO
    :param describe: boolean flag indicating whether descriptive statistics will be displayed.
    :param plot: boolean flag indicating whether plotting mode is enabled.
    :param x: feature name assigned to the x-axis, or None if not specified.
    :param y: feature name assigned to the y-axis, or None if not specified.
    :param z: feature name assigned to the z-axis, or None if not specified.

    :return: None
    """
    cli_console = Console()
    config_table = Table()
    config_table.add_column("Attribute")
    config_table.add_column("Value")

    x = "N/A" if x is None else x
    y = "N/A" if y is None else y
    z = "N/A" if z is None else z
    nm = "N/A" if normalize is None else NORM_LABEL[normalize]

    config_table.add_row("Dataset", dataset)
    config_table.add_row("k", str(k))
    config_table.add_row("Query Datapoint", str(query_pt))
    config_table.add_row("Distance Metric", DISTANCE_LABEL[distance])
    config_table.add_row("Normalization Method", nm)
    config_table.add_row("Enable Descriptive Statistics?", str(describe))
    config_table.add_row("Enable Plotting?", str(plot))
    config_table.add_row("Plot Label for x-axis", x)
    config_table.add_row("Plot Label for y-axis", y)
    config_table.add_row("Plot Label for z-axis", z)

    cli_console.print(config_table, justify="center")

def main(
    dataset: str = typer.Argument("-d", help="a comma separated training data file."),
    k: int = typer.Argument("-k", help="the number of nearest neighbors to be considered."),
    query_data: Annotated[str, typer.Option("--p", help="the query data point's feature values separated by whitespace.")] = None,
    distance: Annotated[Distances, typer.Option("--m", help = "the distance metric", case_sensitive = False)] = Distances.eucl,
    normalize: Annotated[NormalizationMethods, typer.Option("--normalize", help="normalizes feature values using z-scores/min-max scaling.",)] = None,
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
    :param describe: if True, displays a descriptive statistics table for the dataset.
    :param plot: if True, enables scatter plot generation.
    :param x: feature name to plot on the x-axis. Requires --plot.
    :param y: feature name to plot on the y-axis. Requires --x.
    :param z: feature name to plot on the z-axis. Produces a 3D plot. Requires --y.
    :param confirm: if True, displays a configuration summary and prompts the user to proceed.
    :param normalize: #TODO

    :return: None
    """
    console = Console()
    try:
        validate_prediction_args(dataset, k, normalize)
        query_point = get_valid_query_point(query_data)
        user_datapoints, feature_map = load_dataset(dataset)
        validate_dataset_args(user_datapoints, feature_map, k, query_data, plot, x, y, z)

        categories = sorted({pt.category for pt in user_datapoints})

        if confirm:
            console.rule("[bold cyan] Configuration Attributes")
            display_config(dataset, k, query_point, distance, normalize, describe, plot, x, y, z)

            if not typer.confirm("Proceed?"):
                raise typer.Exit()

        column_values = get_column_values(user_datapoints, feature_map)
        mean_of_data = mean_dataset(column_values)
        st_devs = standard_deviation_dataset(column_values)
        data_count, data_min, data_max = count_min_max(column_values)

        normalized_values = None
        normalized_pt = None

        if not normalize:
            distances = calculate_distances(query_point, user_datapoints, distance)
            k_nearest_dists = k_nearest_points(k, distances)
            query_data_prediction = get_classification(k_nearest_dists)

        else:
            if normalize == NormalizationMethods.zscore:
                mean_std_map = get_mean_std_map(mean_of_data, st_devs)
                normalized_values, normalized_pt = normalized_values_zscore(user_datapoints, feature_map,
                                                                            mean_std_map, query_point)

            elif normalize == NormalizationMethods.minmax:
                min_max_map = get_min_max_map(data_min, data_max)
                normalized_values, normalized_pt = normalized_values_minmax(user_datapoints, feature_map,
                                                                            min_max_map, query_point)

            normalized_user_points = get_normalized_datapoints(user_datapoints, normalized_values, feature_map)
            normalized_distances = calculate_distances(normalized_pt, normalized_user_points, distance)
            k_nearest_dists = k_nearest_points(k, normalized_distances)
            query_data_prediction = get_classification(k_nearest_dists)

        console.print(
            Panel(
                Align.center(f"[bold green]{query_data_prediction}[/bold green]"),
                title="Prediction",
                border_style="green"
            )
        )

        console.rule("[bold cyan]Dataset Summary")
        print("Number of features:", len(feature_map))
        cat_str = ""
        for i in range(len(categories)):
            ct = categories[i][0].upper() + categories[i][1:].lower()

            if i == len(categories) - 1:
                cat_str += ct
            else:
                cat_str += ct + ", "

        print("Categories:", cat_str)
        print()

        if describe:
            console.rule("[bold cyan]Descriptive Statistics")
            median_of_data = median_dataset(column_values)
            q1, q3 = quartile_values_dataset(column_values)
            generate_desc_statistics(mean_of_data, data_count, data_min, data_max, q1, median_of_data, q3,
                                     st_devs)

        if plot:
            generate_plots(user_datapoints, feature_map, k, query_point, x, y, z)

    except ValueError as e:
        print(e)
        raise typer.Exit(code = 1)

if __name__ == '__main__':
    typer.run(main)