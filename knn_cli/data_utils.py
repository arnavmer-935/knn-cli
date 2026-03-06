import os.path
import csv
from math import isnan
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


def args_valid(dataset: str, k: int, query_data, x, y, z):
    if k <= 0:
        raise ValueError("The value of k must be positive.")

    if not os.path.isfile(dataset):
        raise ValueError(f"Dataset file: {dataset} is invalid.")

    with open(dataset, newline='') as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("Dataset is empty or malformed.")

        column_count = len(reader.fieldnames)
        if column_count <= 2:
            raise ValueError("Insufficient columns in dataset file. Expected at least 3 columns.")

        query_pt = query_data.strip().split()
        if len(query_pt) == 0:
            raise ValueError("Query datapoint is empty.")

        if column_count - 1 != len(query_pt):
            raise ValueError("Number of feature columns in dataset does not match the dimensions of query point.")

        for val in query_pt:
            try:
                numeric = float(val)
            except ValueError:
                raise ValueError("Query datapoint contains non-numerical data.")

        linecount = 0
        for _ in reader:
            linecount += 1

        if k > linecount:
            raise ValueError("Value of k cannot exceed the size of dataset.")

        feature_set = set(reader.fieldnames)
        if x is not None:
            if x not in feature_set:
                raise ValueError(f"Feature column {x} does not exist in dataset.")

        if y is not None:
            if y not in feature_set:
                raise ValueError(f"Feature column {y} does not exist in dataset.")

        if z is not None:
            if z not in feature_set:
                raise ValueError(f"Feature column {z} does not exist in dataset.")
