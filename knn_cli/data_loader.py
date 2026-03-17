from knn_cli.data_utils import Datapoint
import csv

def load_dataset(dataset: str):
    """
    Parses and processes an existent CSV file containing numeric values.
    :param dataset: the file path of data

    :return: a tuple containing a list of datapoint objects and a dictionary which maps each column name to a 0-based
    index. Used for quickly accessing feature values associated with the given column name.
    """
    datapoints = []
    feature_index_map = dict()
    try:
        with open(dataset, newline='') as file:
            reader = csv.DictReader(file)
            for i in range(len(reader.fieldnames) - 1):
                feature_index_map[reader.fieldnames[i]] = i

            for row in reader:
                if not row:
                    continue

                values = list(row.values())
                feature_vals = [float(val.strip()) for val in values[:-1]]
                category = values[-1].strip()
                datapoint = Datapoint(tuple(feature_vals), category)
                datapoints.append(datapoint)

    except FileNotFoundError:
        print(f"Error: {dataset} does not exist.")

    return datapoints, feature_index_map