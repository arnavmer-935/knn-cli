from knn_cli.data_utils import Datapoint
import csv

def load_dataset(dataset: str, categorical_label: str) -> tuple[list[Datapoint], dict[str, int]]:
    """
    Parses an existing CSV file containing numeric feature values and a categorical label column.
    Assumes that the last column is the category and all preceding columns are numeric features.
    If the file is not found, prints an error message and returns empty results silently.

    :param categorical_label: #TODO
    :param dataset: file path of the training dataset.

    :return: a tuple containing a list of Datapoint objects and a dictionary mapping
    each feature column name to its 0-based index.
    """
    datapoints = []
    feature_index_map = dict()
    with open(dataset, newline='') as file:
        reader = csv.DictReader(file)
        reader.fieldnames = [col.strip() for col in reader.fieldnames]
        cat_idx = reader.fieldnames.index(categorical_label)
        seq_idx = 0
        for i in range(len(reader.fieldnames)):
            if i != cat_idx:
                feature_index_map[reader.fieldnames[i].strip()] = seq_idx
                seq_idx += 1

        for row in reader:
            if not row:
                continue

            values = list(row.values())
            try:
                feature_vals = [float(values[i].strip()) for i in range(len(values)) if i != cat_idx]
            except ValueError:
                raise ValueError("Dataset contains non-numeric values in feature columns.")

            category = values[cat_idx].strip()
            datapoint = Datapoint(tuple(feature_vals), category)
            datapoints.append(datapoint)

    return datapoints, feature_index_map

def get_column_names(dataset):
    with open(dataset, newline='') as f:
        return [col.strip() for col in csv.DictReader(f).fieldnames]