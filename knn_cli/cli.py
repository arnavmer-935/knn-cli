import typer

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from knn_cli.data_utils import get_valid_dist_metric, get_valid_plot_args, get_valid_tts_fraction, display_column_names, \
    ERR_COLOR
from knn_cli.data_loader import load_dataset, get_column_names
from knn_cli.knn import calculate_distances, get_classification, k_nearest_points
from knn_cli.visualization import generate_plots

from knn_cli.data_utils import (
                                get_column_values,
                                NormalizationMethods, KNNConfig, NORM_LABEL,
                                DISTANCE_LABEL, DescriptiveStats, get_format_color, Computation,
                                get_improvement_interpretation, get_valid_dataset_path, get_valid_k,
                                get_valid_categorical_label, get_normalization_requirement, get_model_pathway,
                                get_query_input,
                                landing_message
                                )

from knn_cli.statistics import (mean_dataset,
                                median_dataset,
                                count_min_max,
                                quartile_values_dataset,
                                standard_deviation_dataset,
                                generate_desc_statistics
                                )

from knn_cli.normalization import get_mean_std_map, get_min_max_map, \
    get_normalized_datapoints, normalize_query_point_minmax, normalize_dataset_zscore, normalize_query_point_zscore, \
    normalize_dataset_minmax
from knn_cli.train_test_splitting import train_test_split, get_accuracy, get_baseline_accuracy


def display_config(config: KNNConfig) -> None:
    """
    Displays a formatted table summarizing the user's configuration before execution.
    Called only when the --confirm flag is set, giving the user a chance to review
    all arguments prior to running the classification algorithm.

    :param config: the instance of the dataclass which stores all command line argument inputs from the user

    :return: None
    """
    cli_console = Console()
    config_table = Table()
    config_table.add_column("Attribute")
    config_table.add_column("Value")

    x = "N/A" if config.x is None else config.x
    y = "N/A" if config.y is None else config.y
    z = "N/A" if config.z is None else config.z
    nm = "N/A" if config.normalize is None else NORM_LABEL[config.normalize]
    query = "N/A" if config.query_pt is None else config.query_pt

    enabled_plot = "Yes" if config.plot else "No"
    desc_stats = "Yes" if config.describe else "No"

    tts_msg = None
    if config.tts is None:
        tts_msg = "N/A"

    else:
        training_percent = "N/A" if config.tts is None else f"{(1 - config.tts) * 100:.2f}%"
        testing_percent = "N/A" if config.tts is None else f"{config.tts * 100:.2f}%"
        tts_msg = f"{training_percent} training, {testing_percent} testing"

    config_table.add_row("Dataset", config.dataset)
    config_table.add_row("k", str(config.k))
    config_table.add_row("Query Datapoint", str(query))
    config_table.add_row("Distance Metric", DISTANCE_LABEL[config.distance])
    config_table.add_row("Train-Test split", tts_msg)
    config_table.add_row("Normalization Method", nm)
    config_table.add_row("Enable Descriptive Statistics?", desc_stats)
    config_table.add_row("Enable Plotting?", enabled_plot)
    config_table.add_row("Plot Label for x-axis", x)
    config_table.add_row("Plot Label for y-axis", y)
    config_table.add_row("Plot Label for z-axis", z)

    cli_console.print(config_table, justify="center")

def main() -> None:

    """
    The main entry point of the CLI. Responsible for parsing command line arguments,
    validating inputs, classifying the query point using the KNN algorithm, and
    optionally displaying descriptive statistics and scatter plots.

    Catches and reports ValueError for all invalid argument combinations, including
    invalid file paths, out-of-range k values, query point dimension mismatches,
    and malformed plot configurations. Exits with code 1 on any validation failure.

    :return: None
    """

    console = Console()
    try:
        landing_message(console)

        console.rule("[bold cyan] Dataset Configuration")
        dataset = get_valid_dataset_path()
        user_dataset_columns = get_column_names(dataset)

        display_column_names(console, user_dataset_columns)

        while True:
            categorical_label = get_valid_categorical_label(user_dataset_columns)
            try:
                user_datapoints, feature_map = load_dataset(dataset, categorical_label)
                break
            except ValueError:
                typer.echo(typer.style(
                    "Dataset contains non-numeric feature values with this label column. Please choose a different one.",
                    fg=ERR_COLOR
                ))

        k = get_valid_k(user_datapoints)
        distance = get_valid_dist_metric()

        column_values = get_column_values(user_datapoints, feature_map)
        mean_of_data = mean_dataset(column_values)
        st_devs = standard_deviation_dataset(column_values)
        data_count, data_min, data_max = count_min_max(column_values)

        normalize = get_normalization_requirement()

        query_point = None
        plot, x, y, z = None, None, None, None
        tts = None
        normalized_values = None
        normalized_pt = None

        pathway = get_model_pathway()
        if pathway == "classification":
            query_point = get_query_input(feature_map)
            plot, x, y, z = get_valid_plot_args(feature_map)
            tts = None

        elif pathway == "evaluation":
            tts = get_valid_tts_fraction()

        if normalize == NormalizationMethods.zscore:
            mean_std_map = get_mean_std_map(mean_of_data, st_devs)
            normalized_values = normalize_dataset_zscore(user_datapoints, feature_map, mean_std_map)
            normalized_pt = normalize_query_point_zscore(feature_map, mean_std_map, query_point)

        elif normalize == NormalizationMethods.minmax:
            min_max_map = get_min_max_map(data_min, data_max)
            normalized_values = normalize_dataset_minmax(user_datapoints, feature_map, min_max_map)
            normalized_pt = normalize_query_point_minmax(feature_map, min_max_map, query_point)

        normalized_user_points = None
        if normalized_values is not None:
            normalized_user_points = get_normalized_datapoints(user_datapoints, normalized_values, feature_map)

        typer.echo(" ")
        confirm = typer.confirm("Generate configuration summary for confirmation?")
        typer.echo(" ")
        desc = typer.confirm("Generate descriptive statistics for dataset?")

        categories = sorted({pt.category for pt in user_datapoints})

        user_config = KNNConfig(dataset, k, query_point, categories, distance, normalize, desc, plot, x, y, z, tts)

        user_computation_config = Computation(
            user_datapoints,
            query_point,
            column_values,
            feature_map,
            normalized_user_points,
            normalized_pt,
            mean_of_data,
            st_devs,
            data_count,
            data_min,
            data_max
        )

        if confirm:
            console.rule("[bold cyan] Configuration Attributes")
            display_config(user_config)

            if not typer.confirm("Proceed?"):
                raise typer.Exit()

        if desc:
            console.rule("[bold cyan]Descriptive Statistics")
            median_of_data = median_dataset(column_values)
            q1, q3 = quartile_values_dataset(column_values)

            stats_config = DescriptiveStats(
                mean_of_data,
                data_count,
                data_min,
                data_max,
                median_of_data,
                q1, q3,
                st_devs
            )

            generate_desc_statistics(stats_config)

        if pathway == "classification":
            classification_and_analysis(console, user_config, user_computation_config)
        elif pathway == "evaluation":
            evaluation(console, user_config, user_computation_config)

    except ValueError as e:
        print(e)
        raise typer.Exit(code=1)

def classification_and_analysis(console: Console, knn_config: KNNConfig, computation: Computation):
    """
    Performs KNN classification for a query point and displays the results along with dataset metadata.

    This function computes distances between the query point and dataset points, selects the k-nearest
    neighbors, and determines the predicted class. If normalized data is available, it uses the normalized
    values; otherwise, it falls back to the original data.

    The prediction is displayed using a formatted console panel. Additionally, the function prints a
    summary of the dataset, including the number of features and available categories. Optionally,
    visualization plots are generated if enabled in the configuration.

    :param console: Rich Console object used for formatted output display.
    :param knn_config: Configuration object containing KNN parameters such as k, distance metric,
                       query point, categories, and plotting preferences.
    :param computation: an instance of the Computation dataclass, containing datapoints, normalized
                        datapoints (if not None), query point (normalized if flag entered by user), and feature mappings.

    :return: None
    """
    requirements = computation.normalized_query, computation.normalized_datapoints
    query_data_prediction = None

    if any(x is None for x in requirements):
        distances = calculate_distances(knn_config.query_pt, computation.datapoints, knn_config.distance)
        k_nearest_dists = k_nearest_points(knn_config.k, distances)
        query_data_prediction = get_classification(k_nearest_dists)

    else:
        normalized_distances = calculate_distances(requirements[0], requirements[1], knn_config.distance)
        k_nearest_dists = k_nearest_points(knn_config.k, normalized_distances)
        query_data_prediction = get_classification(k_nearest_dists)

    console.print(
        Panel(
            Align.center(f"[bold green]{query_data_prediction}[/bold green]"),
            title="Prediction",
            border_style="green"
        )
    )

    console.rule("[bold cyan]Dataset Summary")
    print("Number of features:", len(computation.feature_map))
    categories = knn_config.categories
    cat_str = ""
    for i in range(len(categories)):
        ct = categories[i][0].upper() + categories[i][1:].lower()

        if i == len(categories) - 1:
            cat_str += ct
        else:
            cat_str += ct + ", "

    print("Categories:", cat_str)
    print()

    if knn_config.plot:
        generate_plots(computation.datapoints, computation.feature_map, knn_config.k, knn_config.query_pt,
                       knn_config.x, knn_config.y, knn_config.z)


def evaluation(console: Console, config: KNNConfig, computation: Computation):
    """
    Performs model evaluation using a train-test split and displays performance metrics.

    This function splits the dataset into training and testing sets, computes baseline accuracy,
    and evaluates the KNN model's accuracy. If normalization is enabled, the function applies the
    specified normalization method (z-score or min-max) using statistics derived from the training set,
    and evaluates the model on normalized data.

    The function then calculates the improvement of the model over the baseline and displays the
    results in a formatted console panel, including an interpretation of the improvement.

    Note:
    This function is mutually exclusive with classification_and_analysis and should not be invoked
    simultaneously.

    :param console: Rich Console object used for formatted output display.
    :param config: Configuration object containing KNN parameters such as k, distance metric,
                   train-test split ratio, normalization method, and other evaluation settings.
    :param computation: Object containing dataset-related data, including datapoints and feature mappings.

    :return: None
    """
    data_feature_map = computation.feature_map
    training, testing = train_test_split(computation.datapoints, config.tts)
    training_column_values = get_column_values(training, data_feature_map)

    normalized_training_values = None
    normalized_testing_values = None
    model_baseline_accuracy = None
    model_accuracy = None

    if config.normalize is None:
        model_baseline_accuracy = get_baseline_accuracy(training, testing)
        model_accuracy = get_accuracy(config.k, config.distance, training, testing)

    else:
        if config.normalize == NormalizationMethods.zscore:

            training_mean_map = mean_dataset(training_column_values)
            training_std_map = standard_deviation_dataset(training_column_values)

            training_mean_std_map = get_mean_std_map(training_mean_map, training_std_map)

            normalized_training_values = normalize_dataset_zscore(training, data_feature_map, training_mean_std_map)
            normalized_testing_values = normalize_dataset_zscore(testing, data_feature_map, training_mean_std_map)

        elif config.normalize == NormalizationMethods.minmax:
            _, training_min_map, training_max_map = count_min_max(training_column_values)

            min_max_map = get_min_max_map(training_min_map, training_max_map)
            normalized_training_values = normalize_dataset_minmax(training, data_feature_map, min_max_map)
            normalized_testing_values = normalize_dataset_minmax(testing, data_feature_map, min_max_map)

        normalized_training_points = get_normalized_datapoints(training, normalized_training_values, data_feature_map)
        normalized_testing_points = get_normalized_datapoints(testing, normalized_testing_values, data_feature_map)

        model_accuracy = get_accuracy(config.k, config.distance, normalized_training_points, normalized_testing_points)
        model_baseline_accuracy = get_baseline_accuracy(normalized_training_points, normalized_testing_points)


    improvement = model_accuracy - model_baseline_accuracy
    disp_color = get_format_color(improvement)
    disp_interpret = get_improvement_interpretation(improvement)
    baseline = f"{model_baseline_accuracy:.2%}"
    accuracy = f"{model_accuracy:.2%}"
    imp = f"{improvement:+.2%}"

    content = "\n" + (
        f"Baseline Accuracy: {baseline}\n"
        f"Model Accuracy: {accuracy}\n"
        f"[bold {disp_color}]Accuracy Improvement: {imp}[/bold {disp_color}]\n"
        f"[bold {disp_color}]What this means: {disp_interpret}[/bold {disp_color}]"
    ) + "\n"
    console.print(
        Panel(
            Align.center(content),
            title="Accuracy Metrics",
            border_style=f"{disp_color}"
        )
    )

if __name__ == '__main__':
    typer.run(main)