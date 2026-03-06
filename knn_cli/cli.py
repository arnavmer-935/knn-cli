import typer
from typing import Annotated

from knn_cli.data_loader import load_dataset
from knn_cli.data_utils import get_categories, Distances, validate_args
from knn_cli.knn import calculate_distances, get_classification, k_nearest_points
from knn_cli.visualization import generate_plots
from knn_cli.statistics import (mean_dataset, median_dataset, count_min_max, quartile_values_dataset, standard_deviation_dataset,
                         generate_desc_statistics)

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
):

    """
    The main function of the program responsible for setting up the CLI, parsing command line arguments, classifying
    query data based on the KNN algorithm, providing descriptive statistics about the data, and generating plots to
    visualize the data.
    Also handles the following issues:
    - FileNotFound Errors
    - Missing x or y values when plot mode is enabled
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
    try:
        validate_args(dataset, k, query_data, x, y, z)

        user_datapoints, feature_map = load_dataset(dataset)

        categories = get_categories(dataset)

        print("KNN Classifier!")
        print("Training data:", dataset)

        print("Number of features:", len(feature_map))
        print("Categories:")

        for cat in categories[:-1]:
            print("\t" + cat[0].upper() + cat[1:].lower())

        query_point = [float(x) for x in query_data.strip().split()]

        print()
        print(f"Considering {k} nearest neighbors")
        print("Distance Metric:", distance.value)
        print("Query data point:", query_data)

        distances = calculate_distances(query_point, user_datapoints, distance)
        k_nearest_dists = k_nearest_points(k, distances)
        query_data_prediction = get_classification(k_nearest_dists)

        print("Prediction:", query_data_prediction)

        if describe:
            mean_of_data = mean_dataset(dataset)
            median_of_data = median_dataset(dataset)
            data_count, data_min, data_max = count_min_max(dataset)
            q1, q3 = quartile_values_dataset(dataset)
            st_devs = standard_deviation_dataset(dataset)

            generate_desc_statistics(describe, mean_of_data, data_count, data_min, data_max, q1, median_of_data, q3,
                                     st_devs)

        if plot:
            generate_plots(dataset, k, query_data, x, y, z)

    except ValueError as e:
        print(e)
        raise typer.Exit(code = 1)

if __name__ == '__main__':
    typer.run(main)