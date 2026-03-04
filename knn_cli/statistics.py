from math import sqrt
from rich.console import Console
from rich.table import Table
from knn_cli.data_utils import get_column_values, median

def generate_desc_statistics(describe, mean_of_data, ct, min_v, max_v, q1, medians, q3, std):
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


def mean_dataset(dataset):
    """
    Calculates the average values of each independent variable.
    :param dataset: file path of data
    :return: dictionary whose key is the variable name and value is its average
    """
    kv_pair = get_column_values(dataset)
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
    var_values = get_column_values(dataset)
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
    data = get_column_values(dataset)
    var_q1_map = {}
    var_q3_map = {}
    for k in data:
        first_quartile_data = [x for x in data[k] if x < median(data[k])]
        third_quartile_data = [y for y in data[k] if y > median(data[k])]
        var_q1_map[k] = median(first_quartile_data)
        var_q3_map[k] = median(third_quartile_data)

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
    vals = get_column_values(dataset)
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
    d = get_column_values(dataset)
    squared_diffs = []
    var_std_map = {}
    for variable in d:
        var_mean = sum(d[variable]) / len(d[variable])
        for val in d[variable]:
            squared_diffs.append((var_mean - val) ** 2)
        var_std_map[variable] = sqrt(sum(squared_diffs) / len(squared_diffs))
        squared_diffs = []

    return var_std_map