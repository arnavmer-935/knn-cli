from .data_utils import Datapoint

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