from knn_cli.data_utils import Datapoint
import csv

def load_dataset(dataset: str) -> tuple[list[Datapoint], dict[str, int]]:
    """
    Parses an existing CSV file containing numeric feature values and a categorical label column.
    Assumes that the last column is the category and all preceding columns are numeric features.
    If the file is not found, prints an error message and returns empty results silently.

    :param dataset: file path of the training dataset.

    :return: a tuple containing a list of Datapoint objects and a dictionary mapping
    each feature column name to its 0-based index.
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