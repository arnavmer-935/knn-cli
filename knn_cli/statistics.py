from math import sqrt
from rich.console import Console
from rich.table import Table
from knn_cli.data_utils import median

def generate_desc_statistics(mean_of_data, ct, min_v, max_v, q1, medians, q3, std):
    """
    The method responsible for generating a table containing descriptive statistics about the dataset.
    It produces the counts, mean, min and max values, quartile values,
    and standard deviation for each independent variable's column. 
    The table does not contain the categorical values (dependent variable) when 
    descriptive statistics are being reported.
     
    :param mean_of_data: A dictionary where the key is the column name, 
    and the value is the average of that column's values
    
    :param ct: The count (number of data values) for each column in the dataset.
    :param min_v: A dictionary which maps a column name to its minimum value. 
    :param max_v: A dictionary which maps a column name to its maximum value.
    :param q1: A dictionary which maps a column name to the value at its 25th percentile.
    :param medians: A dictionary which maps a column name to its median value (50th percentile).
    :param q3: A dictionary which maps a column name to the value at its 75th percentile.
    :param std: A dictionary which maps a column name to the value of its standard deviation.
    
    :return: None
    """
    console = Console()
    stats_table = Table("Feature", "Count", "Min", "Max", "25%", "50%", "75%",
                        "Mean", "Std\nDev", title_justify="center")
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

    console.print(stats_table, justify="center")

def mean_dataset(column_values):
    """
    Calculates the average values of each independent variable.
    :param column_values: a dictionary which maps each column name to a list of its dataset values.
    :return: dictionary that maps the column name to its average value.
    """
    var_mean_map ={}
    for key in column_values:
        var_mean_map[key] = round(sum(column_values[key]) / len(column_values[key]), 2)

    return var_mean_map

def median_dataset(column_values):
    """
    Calculates the median value (50th percentile) of each independent variable
    :param column_values: a dictionary which maps each column name to a list of its dataset values.
    :return: dictionary that maps the column name to its median value.
    """
    var_median_map = {}
    for key in column_values:
        n = len(column_values[key])
        column_values[key] = sorted(column_values[key])
        if n % 2 != 0:
            var_median_map[key] = column_values[key][n//2]
        else:
            var_median_map[key] = (column_values[key][n//2] + column_values[key][n//2 - 1]) / 2

    return var_median_map

def quartile_values_dataset(column_values):
    """
    Calculates the 25th percentile and 75th percentile values for each independent variable.
    :param column_values: a dictionary which maps each column name to a list of its dataset values.
    :return: tuple of two dictionaries with the following structure:
    a map of column name and 25th percentile values,
    and a map of column name and 75th percentile values.
    """
    var_q1_map = {}
    var_q3_map = {}
    for k in column_values:
        first_quartile_data = [x for x in column_values[k] if x < median(column_values[k])]
        third_quartile_data = [y for y in column_values[k] if y > median(column_values[k])]
        var_q1_map[k] = median(first_quartile_data)
        var_q3_map[k] = median(third_quartile_data)

    return var_q1_map, var_q3_map

def count_min_max(column_values):
    """
    Calculates the count, minimum value and maximum value of each independent variable.
    :param column_values: a dictionary which maps each column name to a list of its dataset values.
    :return: tuple containing count, dictionary that maps column name to is its min value, and
    a dictionary that maps column name to its max value.
    """

    count = len(next(iter(column_values.values())))
    min_map = {}
    max_map = {}
    for var in column_values:
        min_map[var] = min(column_values[var])
        max_map[var] = max(column_values[var])
    return count, min_map, max_map

def standard_deviation_dataset(column_values):
    """
    Calculates the standard deviation about the mean for each independent variable.
    :param column_values: a dictionary which maps each column name to a list of its dataset values.
    :return: dictionary that maps column names to their standard deviation values.
    """
    squared_diffs = []
    var_std_map = {}
    for variable in column_values:
        var_mean = sum(column_values[variable]) / len(column_values[variable])
        for val in column_values[variable]:
            squared_diffs.append((var_mean - val) ** 2)

        var_std_map[variable] = sqrt(sum(squared_diffs) / len(squared_diffs))
        squared_diffs = []

    return var_std_map