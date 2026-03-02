import typer
from dataclasses import dataclass
from typing_extensions import Annotated
from typing import List
from enum import Enum
from distance_metric import *
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
from random import *

color_palette = ["red", "green", "blue", "cyan", "magenta", "yellow", "black", "white", "orange", "purple", "brown",
                 "pink", "gray", "olive", "teal", "navy", "gold", "lime", "indigo", "turquoise"]

@dataclass
class Datapoint:
    features: tuple[float,...]
    category: str

class Distances(str, Enum):
    eucl = "eucl"
    manh = "manh"
    cos = "cos"

def load_dataset(dataset: str):
    """
    Takes a data's file path, processes it and returns a list of datapoint objects.
    Also handles FileNotFound exceptions
    :param dataset: the file path of data
    :return: list of datapoint objects
    """
    datapoints = []
    try:
        with open(dataset, encoding='utf-8') as data:
            lines = data.readlines() # Skip first 5 lines
            for line in lines:
                line = line.strip()
                if not line or "#" in line or not line[0].isdigit():
                    continue
                else:
                    record = line.strip().split(",")
                    feature_vals = [float(val) for val in record[:-1]]
                    category = record[-1]
                    datapoint = Datapoint(tuple(feature_vals), category)
                    datapoints.append(datapoint)
    except FileNotFoundError:
        print(f"Error: {dataset} does not exist!")
    return datapoints

def calculate_distances(feature: list[float], datapoints: list[Datapoint], distance: str = "eucl"): #feature = query datapoint
    """
    Takes the query data given by the user and calculates its distances from the other datapoints. Distance calculation
    method is also chosen by the user, but the default value is Euclidean distance.
    Returns a dictionary whose key is the datapoint in the training data, and the value is the distance between that
    training datapoint and query datapoint.
    :param feature: query datapoint entered by user
    :param datapoints: list of Datapoint Objects
    :param distance: the distance metric chosen by the user
    :return: dictionary whose key is the datapoint in the training data, and the value is the distance between that
    training datapoint and query datapoint.
    """
    feature_distance_map = {}
    if distance == "manh":
        for point in datapoints:
            feature_distance_map[point.features] = manhattan(feature, list(point.features))
    elif distance == "cos":
        for point in datapoints:
            feature_distance_map[point.features] = cosine(feature, list(point.features))
    else:
        for point in datapoints:
            feature_distance_map[point.features] = euclidean(feature, list(point.features))
    return [feature_distance_map, list(feature_distance_map.values())]
    #i = 0: dict key is tuple(feature values), value is distance from user's entered point
    #i = 1: list containing all distances from user's feature point

def _find_min(distances: list[float]) -> int:
    """
    Helper method used to find the index of the smallest value in an array of floating point numbers.
    :param distances: an array containing the distances between the query data and the training data points
    :return: index at which smallest distance is located
    """
    min_index = 0
    for i in range(1, len(distances)):
        if distances[i] < distances[min_index]:
            min_index = i
    return min_index

def k_nearest_distances(k: int, distances: list[float]) -> list[float]:
    """
    Finds the k smallest distances from the array of all distances. The value of k is given by the user.
    Returns a sub-array containing the smallest distance.
    :param k: number of neighbors with which comparison has to be made
    :param distances: an array containing the distances between the query data and the training data points
    :return: a sub-array pof distances which has the k smallest distances
    """
    nearest_distances = []
    while len(nearest_distances) < k:
        min_distance = distances.pop(_find_min(distances))
        nearest_distances.append(min_distance)
    return nearest_distances

def get_classification(nearest_distances: list[float], feature_distance_map, dataset_points: list[Datapoint]):
    #1. USe nearest distances to get feature points of data
    #2.0 Find category for each feature point using the feature_distance_map
    #2.1 Find the category with most appearances
    #3. Return that category
    """
    Uses the k nearest distances, a dictionary whose key is a training datapoint and value is distance from the point,
    and a list of Datapoint objects to find the categories of the nearest neighbors.
    Returns an array containing the k nearest neighbors' categories
    :param nearest_distances: the k smallest distances between the query datapoint and training data datapoint.
    :param feature_distance_map: a dictionary whose key is a datapoint in the training data, and value is the distance
    between that point and query datapoint.
    :param dataset_points: the list of datapoint objects obtained from loading the data
    :return: the prediction of the query datapoint based on the KNN algorithm
    """
    k_nearest_datapoints = [] #contains feature points of the k nearest neighbors
    k_nearest_neighbors = [] #contains the categories of the k nearest neighbors
    for feature in feature_distance_map:
        if feature_distance_map[feature] in nearest_distances:
            k_nearest_datapoints.append(feature)
        else:
            continue

    for pt in k_nearest_datapoints:
        for point in dataset_points:
            if point.features == pt:
                k_nearest_neighbors.append(point.category)
            else:
                continue

    knn_hashtable = {}
    for category in k_nearest_neighbors:
        if category not in knn_hashtable:
            knn_hashtable[category] = 1
        else:
            knn_hashtable[category] += 1

    classification = k_nearest_neighbors[0] #chooses the first category which appears in the knn list
    for c in knn_hashtable:
        if knn_hashtable[c] > knn_hashtable[classification]:
            classification = c
        else:
            continue
    return classification

def _get_columns(dataset):
    """
    Helper method for getting all the column names of a data.
    :param dataset: file path for data
    :return: list containing the column names
    """
    with open(dataset) as f:
        cols = f.readlines()[0].strip().split(",")
        return cols

def _get_column_values(dataset):
    """
    Helper method for getting all the values of each column.
    :param dataset: file path for data
    :return: a dictionary whose key is the column name and value is a list containing all of its column values.
    """
    column_map = {}
    cols = _get_columns(dataset)
    column_values = [[] for _ in range(len(cols))]
    with open(dataset) as d:
        data = d.readlines()
        for row in data:
            if not row[0].isdigit():
               continue
            else:
                row = row.strip().split(",")
                for i in range(len(row)-1):
                    column_values[i].append(float(row[i]))

        for i in range(len(cols)):
            column_map[cols[i]] = column_values[i]

    return column_map #returns dict with key as i/d variable name and value as all its data in a list

def _median(arr):
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

def mean_dataset(dataset):
    """
    Calculates the average values of each independent variable.
    :param dataset: file path of data
    :return: dictionary whose key is the variable name and value is its average
    """
    kv_pair = _get_column_values(dataset)
    var_mean_map ={}
    for key in kv_pair:
        var_mean_map[key] = round(sum(kv_pair[key]) / len(kv_pair[key]), 2)

    return var_mean_map

def median_dataset(dataset):
    """
    Calculates the median value (50th percentile) of each independent variable
    :param dataset: rfile path of the data
    :return: dictionary whose key is the variable name and the value is its median
    """
    var_values = _get_column_values(dataset)
    var_median_map = {}
    for key in var_values:
        n = len(var_values[key])
        var_values[key] = sorted(var_values[key])
        if n % 2 != 0:
            var_median_map[key] = var_values[key][n//2]
        else:
            var_median_map[key] = (var_values[key][n//2] + var_values[key][n//2 - 1]) / 2

    return var_median_map

def quartile_values_dataset(dataset):
    """
    Calculates the 25th percentile and 75th percentile values for each independent variable.
    :param dataset: file path for data
    :return: a tuple of 2 dictionaries - both have their keys as the variable name, but the value for one is the 25th percentile
    and the other's value is its 75th percentile
    """
    data = _get_column_values(dataset)
    var_q1_map = {}
    var_q3_map = {}
    for k in data:
        m = len(data[k])
        first_quartile_data = [x for x in data[k] if x < _median(data[k])]
        third_quartile_data = [y for y in data[k] if y > _median(data[k])]
        var_q1_map[k] = _median(first_quartile_data)
        var_q3_map[k] = _median(third_quartile_data)

    return var_q1_map, var_q3_map

def count_min_max(dataset):
    """
    Calculates the count, minimum value and maximum value of each independent variable.
    :param dataset: file path for data
    :return: tuple containing count, dict whose key is variable name and value is its min,
    dict whose key is variable name and value is its max
    """
    ls = open(dataset).readlines()[1:] #ignores line containing columns
    filtered_ls = [l for l in ls if not l.strip() or "#" not in l.strip() or l[0].isalpha()]
    c = len(filtered_ls)
    vals = _get_column_values(dataset)
    min_map = {}
    max_map = {}
    for var in vals:
        min_map[var] = min(vals[var])
        max_map[var] = max(vals[var])
    return c, min_map, max_map

def standard_deviation_dataset(dataset):
    """
    Calculates the standard deviation about the mean for each independent variable
    :param dataset: file path for data
    :return: dictionary whose key is the variable name and value is its standard deviation about the mean
    """
    d = _get_column_values(dataset)
    squared_diffs = []
    var_std_map = {}
    for variable in d:
        var_mean = sum(d[variable]) / len(d[variable])
        for val in d[variable]:
            squared_diffs.append((var_mean - val) ** 2)
        var_std_map[variable] = sqrt(sum(squared_diffs) / len(squared_diffs))
        squared_diffs = []

    return var_std_map

def _get_categories(dataset):
    """
    Helper method for getting the different category values in the data
    :param dataset: file path for data
    :return: list of categories
    """
    with open(dataset) as f:
        ls = list(set([l.strip().split(",")[-1] for l in f.readlines()]))
        return ls

def generate_plot(dataset, k, query_data, x, y, plot, user_datapoints, z = None):
    iv = _get_columns(dataset)
    cts = _get_categories(dataset) #e.g ["Setosa", "Virginica", "Versicolor"]
    column_values = _get_column_values(dataset)
    print(column_values)
    used_colors = []
    x_values = []
    y_values = []
    z_values = []
    categories = {x: "" for x in cts}
    if plot:
        if x in iv and y in iv and z is None:
            for p in user_datapoints:
                for i in range(len(x_values)):
                    x_values[i].append(p.features[iv.index(x)])

            for p in user_datapoints:
                for i in range(len(y_values)):
                    y_values[i].append(p.features[iv.index(y)])
            print(x_values)
            print(y_values)
            plt.title(f"KNN Classifier, k = {k}")
            plt.xlabel(x)
            plt.ylabel(y)
            """for i in range(len(x_values)):
                color = choice([c for c in color_palette if c not in used_colors])
                color_list = [color] * len(x_values[i])
                print(color)
                used_colors.append(color)
                plt.scatter(x_values[i], y_values[i], c=color_list, marker='o', edgecolors='green', linewidths=1, s=90)

            plt.scatter(query_data[iv.index(x)], query_data[iv.index(y)], marker="*", s=90)
            plt.show()"""
            plt.scatter(x_values, y_values, c=used_colors, marker="o", edgecolors="green", linewidths=1)
            plt.scatter(query_data[query_data.index(x)], query_data[query_data.index(y)])
            plt.show()
        elif x in iv and y in iv and z in iv:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plt.title(f"KNN Classifier, k = {k}")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_zlabel(z)
            for i in range(len(x_values)):
                new_col = [choice([c for c in color_palette if c not in used_colors])] * len(x_values[i])
                print(new_col)
                used_colors.append(new_col)
                ax.scatter(x_values[i], y_values[i], z_values[i], c=new_col, marker='o', edgecolors='green', linewidths=1,
                           s=90)
            ax.scatter(query_data[iv.index(x)], query_data[iv.index(y)], query_data[iv.index(y)], marker="*", s=90)
            plt.show()
        elif x is None or y is None:
            print("Error: x and y axes labels are required when plot mode is enabled.")
        elif x not in iv:
            print(f"Error: column name {x} not found in the training data.")
        elif y not in iv:
            print(f"Error: column name {y} not found in the training data.")
        else:
            pass
    else:
        pass

def generate_desc_statistics(describe, mean_of_data, ct, min_v, max_v, q1, medians, q3, std):
    if describe:
        console = Console(width=800)
        stats_table = Table("Feature", "Count", "Min", "Max", "25%", "50%", "75%", "Mean", "Std\nDev",
                            title="Descriptive Statistics", title_justify="center")
        for fea in mean_of_data:
            stats_table.add_row(
                fea,
                f"{ct:.2f}",
                f"{min_v[fea]:.2f}",
                f"{max_v[fea]:.2f}",
                f"{q1[fea]:.2f}",
                f"{medians[fea]:.2f}",
                f"{q3[fea]:.2f}",
                f"{mean_of_data[fea]:.2f}",
                f"{std[fea]:.2f}"
            )
        console.print(stats_table)
    else:
        pass

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
    :param x: the value which is to be plotted on the x axis (if plot is enabled)
    :param y: the value which is to be plotted on the y axis (if plot is enabled)
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
        cts = _get_categories(dataset)
        print(cts)
        print("KNN Classifier!")
        print("Training data:", dataset)
        features_of_data = _get_columns(dataset)
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