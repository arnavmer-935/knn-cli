from dataclasses import dataclass
from enum import Enum

@dataclass
class Datapoint:
    features: tuple[float,...]
    category: str

class Distances(str, Enum):
    eucl = "eucl"
    manh = "manh"
    cos = "cos"


def find_min(distances: list[float]) -> int:
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

def get_columns(dataset):
    """
    Helper method for getting all the column names of a data.
    :param dataset: file path for data
    :return: list containing the column names
    """
    with open(dataset) as f:
        cols = f.readlines()[0].strip().split(",")
        return cols

def get_column_values(dataset):
    """
    Helper method for getting all the values of each column.
    :param dataset: file path for data
    :return: a dictionary whose key is the column name and value is a list containing all of its column values.
    """
    column_map = {}
    cols = get_columns(dataset)
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

def median(arr):
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

def get_categories(dataset):
    """
    Helper method for getting the different category values in the data
    :param dataset: file path for data
    :return: list of categories
    """
    with open(dataset) as f:
        ls = list(set([l.strip().split(",")[-1] for l in f.readlines()]))
        return ls