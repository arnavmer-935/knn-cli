#TODO: documentation
# mention that zscore normalization yields zero if std = 0,
# and that minmax scaling yields zero if column is constant (i.e minimum == maximum)

from knn_cli.data_utils import get_column_values, Datapoint

def get_normalized_datapoints(datapoints, normalized_values, feature_map):

    normalized_datapoints = []

    for i in range(len(datapoints)):
        normalized_vals = tuple(normalized_values[feature][i] for feature in feature_map)
        normalized_pt = Datapoint(normalized_vals, datapoints[i].category)
        normalized_datapoints.append(normalized_pt)

    return normalized_datapoints

def normalized_values_zscore(datapoints, feature_map, mean_std_map, query_point):
    column_vals = get_column_values(datapoints, feature_map)
    normalized_values = {}
    normalized_query_point = []

    for col_name, values in column_vals.items():
        mean, std = mean_std_map[col_name]
        normalized_values[col_name] = zscore(values, mean, std)
        idx = feature_map[col_name]

        normalized_pt_value = (query_point[idx] - mean)/std if std != 0 else 0
        normalized_query_point.append(normalized_pt_value)

    return normalized_values, normalized_query_point

def normalized_values_minmax(datapoints, feature_map, min_max_map, query_point):
    column_vals = get_column_values(datapoints, feature_map)
    normalized_values = {}
    normalized_query_point = []

    for col_name, values in column_vals.items():
        min_v, max_v = min_max_map[col_name]
        normalized_values[col_name] = minmax(values, min_v, max_v)
        idx = feature_map[col_name]

        normalized_pt_value = (query_point[idx] - min_v) / (max_v - min_v) if max_v != min_v else 0
        normalized_query_point.append(normalized_pt_value)

    return normalized_values, normalized_query_point

def zscore(values, mean, std):
    res = []
    if std == 0:
        res = [0] * len(values)

    else:
        for val in values:
            res.append((val - mean) / std)

    return res

def minmax(values, min_value, max_value):
    res = []
    if min_value == max_value:
        res = [0] * len(values)

    else:
        for val in values:
            res.append((val - min_value) / (max_value - min_value))

    return res

def get_mean_std_map(mean_map, std_map):
    result = {}

    for feature in mean_map:
        result[feature] = mean_map[feature], std_map[feature]

    return result

def get_min_max_map(min_map, max_map):
    result = {}

    for feature in min_map:
        result[feature] = min_map[feature], max_map[feature]

    return result