from knn_cli.data_utils import Datapoint
import csv

def load_dataset(dataset: str):
    """
    Takes a csv file path, processes it and returns a list of datapoint objects.
    Also handles FileNotFound exceptions
    :param dataset: the file path of data
    :return: list of datapoint objects
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
                feature_vals = [float(val) for val in values[:-1]]
                category = values[-1]
                datapoint = Datapoint(tuple(feature_vals), category)
                datapoints.append(datapoint)


    except FileNotFoundError:
        print(f"Error: {dataset} does not exist.")

    return datapoints, feature_index_map