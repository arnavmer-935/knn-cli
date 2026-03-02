import typer
from typing import List, Annotated
from .data_loader import load_dataset
from .data_utils import get_categories, get_columns, Distances
from .knn import calculate_distances, get_classification, k_nearest_distances
from .visualization import generate_plot
from .statistics import (mean_dataset, median_dataset, count_min_max, quartile_values_dataset, standard_deviation_dataset,
                         generate_desc_statistics)


def main(
    dataset: str = typer.Argument(help="a comma separated training data file."),
    k: int = typer.Argument(help="the number of nearest neighbors to be considered."),
    query_data: List[float] = typer.Argument(help="the query data point's feature values separated by whitespace."),
    distance: Annotated[Distances, typer.Option(help = "the distance metric", case_sensitive = False)] = Distances.eucl,
    describe: Annotated[bool, typer.Option(help="display in the standard output descriptive statistics of the data")] = False,
    plot: Annotated[bool, typer.Option(help="enables plotting mode")] = False,
    x: Annotated[str, typer.Option(help="label for the x axis")] = None,
    y: Annotated[str, typer.Option(help="label for the y axis")] = None,
    z: Annotated[str, typer.Option(help="label for the z axis (used only in 3D plots")] = None
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
        f = open(dataset)
        mean_of_data = mean_dataset(dataset)
        median_of_data = median_dataset(dataset)
        data_count, data_min, data_max = count_min_max(dataset)
        q1, q3 = quartile_values_dataset(dataset)
        st_devs = standard_deviation_dataset(dataset)
        cts = get_categories(dataset)
        print(cts)
        print("KNN Classifier!")
        print("Training data:", dataset)
        features_of_data = get_columns(dataset)
        print("Number of features:", len(features_of_data))
        print("Categories:")
        for c in cts:
            print("\t" + c[0].upper() + c[1:].lower())
        print()
        print(f"Considering {k} nearest neighbors")
        print("Distance:", distance.value)
        print("Query data point:", query_data)

        user_datapoints = load_dataset(dataset)
        user_feature_dist_map = calculate_distances(query_data, user_datapoints, distance.value)[0]
        user_distances_from_feature = calculate_distances(query_data, user_datapoints, distance.value)[1]
        k_nearest_dists = k_nearest_distances(k, user_distances_from_feature)
        query_data_prediction = get_classification(k_nearest_dists, user_feature_dist_map, user_datapoints)

        print("Prediction:", query_data_prediction)

        generate_desc_statistics(describe, mean_of_data, data_count, data_min, data_max, q1, median_of_data, q3, st_devs)

        generate_plot(dataset, k, query_data, x, y, plot, user_datapoints)

    except FileNotFoundError:
        print(f"Error: {dataset} does not exist")

if __name__ == '__main__':
    typer.run(main)