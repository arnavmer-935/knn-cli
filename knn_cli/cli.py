import typer
from typing import Annotated

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from knn_cli.data_loader import load_dataset
from knn_cli.data_utils import Distances, get_column_values, validate_prediction_args, validate_dataset_args
from knn_cli.knn import calculate_distances, get_classification, k_nearest_points
from knn_cli.visualization import generate_plots
from knn_cli.statistics import (mean_dataset, median_dataset, count_min_max, quartile_values_dataset, standard_deviation_dataset,
                         generate_desc_statistics)

def display_config(dataset, k, query_pt, distance, describe, plot, x, y, z):

    cli_console = Console()
    config_table = Table()
    config_table.add_column("Attribute")
    config_table.add_column("Value")

    x = "N/A" if x is None else x
    y = "N/A" if y is None else y
    z = "N/A" if z is None else z
    query_data = [float(x) for x in query_pt.strip().split()]

    config_table.add_row("Dataset", dataset)
    config_table.add_row("k", str(k))
    config_table.add_row("Query Datapoint", str(query_data))
    config_table.add_row("Distance Metric", distance.value)
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
    describe: Annotated[bool, typer.Option(help="display in the standard output descriptive statistics of the data")] = False,
    plot: Annotated[bool, typer.Option(help="enables plotting mode")] = False,
    x: Annotated[str, typer.Option(help="label for the x axis")] = None,
    y: Annotated[str, typer.Option(help="label for the y axis")] = None,
    z: Annotated[str, typer.Option(help="label for the z axis (used only in 3D plots")] = None,
    confirm: Annotated[bool, typer.Option("--confirm", help="confirm arguments before running")] = False
):

    """
    The main function of the program responsible for setting up the CLI, parsing command line arguments, classifying
    query data based on the KNN algorithm, providing descriptive statistics about the data, and generating plots to
    visualize the data.
    Also handles the following issues:
    - FileNotFound Errors
    - Missing x or y values when plot mode is enabled
    :param confirm:
    :param dataset: the file path of data
    :param k: the amount of nearest neighbors required for comparison
    :param query_data: the datapoint whose classification is to be made
    :param distance: the distance metric
    :param describe: flag for choosing whether descriptive statistics for data are needed
    :param plot: flag for choosing whether plots for the data are needed
    :param x: the value which is to be plotted on the x-axis (if plot is enabled)
    :param y: the value which is to be plotted on the y-axis (if plot is enabled)
    :param z: optional value which is to be plotted on the z axis (3D plot is made if z is given, 2D is generated otherwise)
    :return: None
    """
    console = Console()
    try:
        if confirm:
            console.rule("[bold cyan] Configuration Attributes")
            display_config(dataset, k, query_data, distance, describe, plot, x, y, z)

            if not typer.confirm("Proceed?"):
                raise typer.Exit()

        validate_prediction_args(dataset, k, query_data)
        user_datapoints, feature_map = load_dataset(dataset)
        validate_dataset_args(user_datapoints, feature_map, k, query_data, x, y, z)

        categories = sorted({pt.category for pt in user_datapoints})
        query_point = [float(x) for x in query_data.strip().split()]

        distances = calculate_distances(query_point, user_datapoints, distance)
        k_nearest_dists = k_nearest_points(k, distances)
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
            column_values = get_column_values(user_datapoints, feature_map)
            mean_of_data = mean_dataset(column_values)
            median_of_data = median_dataset(column_values)

            data_count, data_min, data_max = count_min_max(column_values)
            q1, q3 = quartile_values_dataset(column_values)
            st_devs = standard_deviation_dataset(column_values)

            generate_desc_statistics(mean_of_data, data_count, data_min, data_max, q1, median_of_data, q3,
                                     st_devs)

        if plot:
            generate_plots(user_datapoints, feature_map, k, query_data, x, y, z)

    except ValueError as e:
        print(e)
        raise typer.Exit(code = 1)

if __name__ == '__main__':
    typer.run(main)