from random import sample
from matplotlib import pyplot as plt
from knn_cli.data_utils import Datapoint

COLOR_PALETTE = ["red","blue","green","orange","purple","brown","pink","gray","olive","cyan","magenta",
                 "teal","navy","coral","lime","indigo","turquoise","maroon","darkgreen","darkblue","darkorange",
                 "slateblue","crimson","peru","dodgerblue","forestgreen","darkviolet","chocolate"]

def generate_plots(datapoints: list[Datapoint], feature_map: dict[str, int], k: int,
                   query_data: list[float], x: str, y: str, z: str) -> None:
    """
    Generates a 2D or 3D scatter plot of the dataset, color-coded by category,
    with the query point highlighted as a star marker.

    A 3D plot is produced when a z-axis feature is provided, otherwise a 2D plot
    is generated. Each category is assigned a distinct color, and the query point
    is always rendered in yellow for visibility.

    :param datapoints: list of Datapoint objects representing the training example_datasets.
    :param feature_map: dictionary mapping each feature name to its 0-based index.
    :param k: number of nearest neighbors used in classification. Displayed in the plot title.
    :param query_data: the parsed query point as a list of floats.
    :param x: feature name to plot on the x-axis.
    :param y: feature name to plot on the y-axis.
    :param z: optional feature name to plot on the z-axis. If provided, a 3D plot
    is generated. If None, a 2D plot is generated instead.

    :return: None
    """
    if x in feature_map and y in feature_map:
        groups = {}
        x_index = feature_map[x]
        y_index = feature_map[y]

        z_index = feature_map[z] if z in feature_map else None

        for point in datapoints:
            if point.category not in groups:
                groups[point.category] = []

            if z_index is not None:
                coordinate = (point.features[x_index], point.features[y_index], point.features[z_index])
            else:
                coordinate = (point.features[x_index], point.features[y_index])

            groups[point.category].append(coordinate)

        if z_index is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111, projection='3d')

        ax.set_title(
            f"KNN Classification (k = {k})",
            fontsize=14,
            fontweight="bold"
        )
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if z_index is not None:
            ax.set_zlabel(z)

        legend = map_colors_to_categories(groups)

        for category_color, category in legend.items():
            plot_points = groups[category]

            x_points = [p[0] for p in plot_points]
            y_points = [p[1] for p in plot_points]
            if z_index is None:
                ax.scatter(x_points, y_points, color = category_color,
                            label = category, marker='o', edgecolors="black",
                            alpha=0.5, linewidths=1, s=90)

            else:
                z_points = [p[2] for p in plot_points]
                ax.scatter(x_points, y_points, z_points, color = category_color,
                            label = category, marker='o', edgecolors="black",
                            alpha=0.5, linewidths=1, s=90)

        if z_index is None:
            ax.scatter(query_data[x_index], query_data[y_index], color="yellow", edgecolors="black", marker="*",
                       s=350, linewidths=2, zorder=10, label="Query Point")
        else:
            ax.scatter(query_data[x_index], query_data[y_index], query_data[z_index], color="yellow",
                       edgecolors="black", marker="*",
                       s=350, linewidths=2, zorder=10, label="Query Point", depthshade="False")

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
            frameon=True
        )

        ax.grid(True, linestyle="--", alpha=0.3)

        plt.show()

def map_colors_to_categories(groups: dict[str, list[tuple[float, float, float]]]) -> dict[str, str]:
    """
    Assigns a unique color to each category in the dataset for use in scatter plots.
    Colors are randomly sampled without replacement from the predefined COLOR_PALETTE.

    :param groups: dictionary mapping each category name to its list of coordinate tuples.
    :return: dictionary mapping each assigned color to its corresponding category name.
    """
    colors = sample(COLOR_PALETTE, len(groups))
    return dict(zip(colors, groups.keys()))
