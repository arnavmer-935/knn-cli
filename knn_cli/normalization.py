from knn_cli.data_utils import get_column_values, Datapoint

def get_normalized_datapoints(datapoints, normalized_values, feature_map):
    """
    Constructs a new list of Datapoint objects using normalized feature values.

    :param datapoints: list of original Datapoint objects.
    :param normalized_values: dictionary mapping feature names to their normalized value lists.
    :param feature_map: dictionary mapping feature names to their 0-based index.

    :return: list of Datapoint objects with normalized feature values, preserving original categories.
    """
    normalized_datapoints = []

    for i in range(len(datapoints)):
        normalized_vals = tuple(normalized_values[feature][i] for feature in feature_map)
        normalized_pt = Datapoint(normalized_vals, datapoints[i].category)
        normalized_datapoints.append(normalized_pt)

    return normalized_datapoints

def normalize_dataset_zscore(datapoints, feature_map, mean_std_map):
    """
    Applies z-score normalization to a list of Datapoints using their computed mean and standard deviations for
    their respective feature values.

    Note: if the standard deviation is 0 (i.e. all values in the column are identical),
    all normalized values are set to 0 to avoid division by zero.

    :param datapoints: list of Datapoints to be normalized.
    :param feature_map: dictionary mapping each feature column name to its 0-based index.
    :param mean_std_map: dictionary mapping each feature column to a tuple containing its mean and standard deviation.

    :return: list of z-score normalized Datapoints.
    """
    column_vals = get_column_values(datapoints, feature_map)
    normalized_values = {}

    for col_name, values in column_vals.items():
        mean, std = mean_std_map[col_name]
        normalized_values[col_name] = zscore(values, mean, std)

    return normalized_values

def normalize_query_point_zscore(feature_map, mean_std_map, query_point):
    """
    Applies z-score normalization to a given query point, using the computed mean and standard deviations for
    its respective feature values.

    :param feature_map: dictionary mapping each feature column name to its 0-based index.
    :param mean_std_map: dictionary mapping each feature column to a tuple containing its mean and standard deviation.
    :param query_point: list of floating point values denoting the query point.

    :return: z-score normalized query point.
    """
    if query_point is None:
        return None

    else:
        normalized_query_point = []
        for col_name in feature_map.keys():
            mean, std = mean_std_map[col_name]
            idx = feature_map[col_name]

            normalized_pt_value = (query_point[idx] - mean) / std if std != 0 else 0
            normalized_query_point.append(normalized_pt_value)

        return normalized_query_point

def normalize_dataset_minmax(datapoints, feature_map, min_max_map):
    """
       Applies min-max scaling to a list of Datapoints using the computed minimums and maximums for
       their respective feature values.

       Note: if the minimum value in a column is equal to its maximum value (i.e. all values in the column are identical),
       all normalized values are set to 0 to avoid division by zero.

       :param datapoints: list of Datapoints to be normalized.
       :param feature_map: dictionary mapping each feature column name to its 0-based index.
       :param min_max_map: dictionary mapping each feature column to a tuple containing its min and max value.

       :return: list of min-max scaled Datapoints.
       """
    column_vals = get_column_values(datapoints, feature_map)
    normalized_values = {}

    for col_name, values in column_vals.items():
        min_v, max_v = min_max_map[col_name]
        normalized_values[col_name] = minmax(values, min_v, max_v)

    return normalized_values

def normalize_query_point_minmax(feature_map, min_max_map, query_point):
    """
    Applies min-max scaling to a given query point, using the computed minimum and maximum column values for
    its respective feature values.

    :param feature_map: dictionary mapping each feature column name to its 0-based index.
    :param min_max_map: dictionary mapping each feature column to a tuple containing its mean and standard deviation.
    :param query_point: list of floating point values denoting the query point.

    :return: min-max scaled query point.
    """
    if query_point is None:
        return None

    else:
        normalized_query_point = []
        for col_name in feature_map.keys():
            min_v, max_v = min_max_map[col_name]
            idx = feature_map[col_name]

            normalized_pt_value = (query_point[idx] - min_v) / (max_v - min_v) if max_v != min_v else 0
            normalized_query_point.append(normalized_pt_value)

        return normalized_query_point

def zscore(values, mean, std):
    """
    Applies z-score normalization to a list of values using the provided mean and standard deviation.

    Note: if the standard deviation is 0 (i.e. all values in the column are identical),
    all normalized values are set to 0 to avoid division by zero.

    :param values: list of floats to normalize.
    :param mean: the mean of the column.
    :param std: the standard deviation of the column.

    :return: list of z-score normalized floats.
    """
    res = []
    if std == 0:
        res = [0] * len(values)

    else:
        for val in values:
            res.append((val - mean) / std)

    return res

def minmax(values, min_value, max_value):
    """
   Applies min-max scaling to a list of values using the provided minimum and maximum.

   Note: if the minimum and maximum are equal (i.e. all values in the column are identical),
   all scaled values are set to 0 to avoid division by zero.

   :param values: list of floats to normalize.
   :param min_value: the minimum value of the column.
   :param max_value: the maximum value of the column.

   :return: list of min-max scaled floats in the range [0, 1].
   """
    res = []
    if min_value == max_value:
        res = [0] * len(values)

    else:
        for val in values:
            res.append((val - min_value) / (max_value - min_value))

    return res

def get_mean_std_map(mean_map, std_map):
    """
    Combines separate mean and standard deviation dictionaries into a single map.

    :param mean_map: dictionary mapping feature names to their mean values.
    :param std_map: dictionary mapping feature names to their standard deviation values.

    :return: dictionary mapping each feature name to a tuple of (mean, std).
    """
    result = {}

    for feature in mean_map:
        result[feature] = mean_map[feature], std_map[feature]

    return result

def get_min_max_map(min_map, max_map):
    """
    Combines separate min and max dictionaries into a single map.

    :param min_map: dictionary mapping feature names to their minimum values.
    :param max_map: dictionary mapping feature names to their maximum values.

    :return: dictionary mapping each feature name to a tuple of (min, max).
    """
    result = {}

    for feature in min_map:
        result[feature] = min_map[feature], max_map[feature]

    return result