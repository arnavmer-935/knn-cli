from data_utils import get_column_values

def normalized_values_zscore(datapoints, feature_map, mean_std_map):
    column_vals = get_column_values(datapoints, feature_map)
    normalized_values = {}

    for col_name, values in column_vals.items():
        mean, std = mean_std_map[col_name]
        normalized_values[col_name] = zscore(values, mean, std)

    return normalized_values

def normalized_values_minmax(datapoints, feature_map, min_max_map):
    column_vals = get_column_values(datapoints, feature_map)
    normalized_values = {}

    for col_name, values in column_vals.items():
        min_v, max_v = min_max_map[col_name]
        normalized_values[col_name] = minmax(values, min_v, max_v)

    return normalized_values

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