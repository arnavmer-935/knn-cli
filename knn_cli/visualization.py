from random import choice
from matplotlib import pyplot as plt

from .data_loader import load_dataset


color_palette = ["red","blue","green","orange","purple","brown","pink","gray","olive","cyan","magenta","gold",
                 "teal","navy","coral","lime","indigo","turquoise","maroon","darkgreen","darkblue","darkorange",
                 "slateblue","crimson","peru","orchid","dodgerblue","forestgreen","darkviolet","chocolate"]


def generate_plots(dataset, k, query_data, x: str, y, plot: bool, z = None):

    datapoints, feature_map = load_dataset(dataset)
    if plot:
        if x in feature_map and y in feature_map:
            groups = {}
            x_index = feature_map[x]
            y_index = feature_map[y]

            if z in feature_map:
                z_index = feature_map[z]
            else:
                z_index = None

            for point in datapoints:
                if point.category not in groups:
                    groups[point.category] = []

                if z_index is not None:
                    coordinate = (point.features[x_index], point.features[y_index], point.features[z_index])
                else:
                    coordinate = (point.features[x_index], point.features[y_index])

                groups[point.category].append(coordinate)

            legend = map_colors_to_categories(groups)
            plt.title(f"KNN Classifier, k = {k}")
            plt.xlabel(x)
            plt.ylabel(y)

            if z_index is None:
                fig, ax = plt.subplots()
            else:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

            ax.set_title(f"KNN Classifier, k = {k}")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            if z_index is not None:
                ax.set_zlabel(z)

            for category_color, category in legend.items(): #key: color, value: category
                plot_points = groups[category]

                x_points = [p[0] for p in plot_points]
                y_points = [p[1] for p in plot_points]
                if z_index is None:
                    ax.scatter(x_points, y_points, color = category_color,
                                label = category, marker='o', edgecolors='black',
                                alpha=0.7, linewidths=1, s=90)

                else:
                    z_points = [p[2] for p in plot_points]
                    ax.scatter(x_points, y_points, z_points, color = category_color,
                                label = category, marker='o', edgecolors='black',
                                alpha=0.7, linewidths=1, s=90)

            ax.legend()
            plt.show()

def map_colors_to_categories(groups):
    used_colors = set()
    colors = []

    for _ in range(len(groups)):
        chosen = choice(color_palette)
        while chosen in used_colors:
            chosen = choice(color_palette)

        colors.append(chosen)
        used_colors.add(chosen)

    return dict(zip(colors, groups.keys()))