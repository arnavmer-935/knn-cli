from random import choice
from matplotlib import pyplot as plt

color_palette = ["red","blue","green","orange","purple","brown","pink","gray","olive","cyan","magenta",
                 "teal","navy","coral","lime","indigo","turquoise","maroon","darkgreen","darkblue","darkorange",
                 "slateblue","crimson","peru","dodgerblue","forestgreen","darkviolet","chocolate"]

def generate_plots(datapoints, feature_map, k, query_data, x, y, z):

    if x in feature_map and y in feature_map:
        groups = {}
        x_index = feature_map[x]
        y_index = feature_map[y]

        z_index = feature_map[z] if z in feature_map else None

        #TODO: takes already valid query point

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

        for category_color, category in legend.items(): #key: color, value: category
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