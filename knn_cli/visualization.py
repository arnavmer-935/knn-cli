from random import choice
from matplotlib import pyplot as plt
from .data_utils import get_columns, get_categories, get_column_values

color_palette = ["red", "green", "blue", "cyan", "magenta", "yellow", "black", "white", "orange", "purple", "brown",
                 "pink", "gray", "olive", "teal", "navy", "gold", "lime", "indigo", "turquoise"]

def generate_plot(dataset, k, query_data, x, y, plot, user_datapoints, z = None):
    iv = get_columns(dataset)
    cts = get_categories(dataset) #e.g ["Setosa", "Virginica", "Versicolor"]
    column_values = get_column_values(dataset)


    print(column_values)
    used_colors = []
    x_values = []
    y_values = []
    z_values = []
    categories = {x: "" for x in cts}
    if plot:
        if x in iv and y in iv and z is None:
            for p in user_datapoints:
                for i in range(len(x_values)):
                    x_values[i].append(p.features[iv.index(x)])

            for p in user_datapoints:
                for i in range(len(y_values)):
                    y_values[i].append(p.features[iv.index(y)])
            print(x_values)
            print(y_values)
            plt.title(f"KNN Classifier, k = {k}")
            plt.xlabel(x)
            plt.ylabel(y)
            """for i in range(len(x_values)):
                color = choice([c for c in color_palette if c not in used_colors])
                color_list = [color] * len(x_values[i])
                print(color)
                used_colors.append(color)
                plt.scatter(x_values[i], y_values[i], c=color_list, marker='o', edgecolors='green', linewidths=1, s=90)

            plt.scatter(query_data[iv.index(x)], query_data[iv.index(y)], marker="*", s=90)
            plt.show()"""
            plt.scatter(x_values, y_values, c=used_colors, marker="o", edgecolors="green", linewidths=1)
            plt.scatter(query_data[query_data.index(x)], query_data[query_data.index(y)])
            plt.show()
        elif x in iv and y in iv and z in iv:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plt.title(f"KNN Classifier, k = {k}")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_zlabel(z)
            for i in range(len(x_values)):
                new_col = [choice([c for c in color_palette if c not in used_colors])] * len(x_values[i])
                print(new_col)
                used_colors.append(new_col)
                ax.scatter(x_values[i], y_values[i], z_values[i], c=new_col, marker='o', edgecolors='green', linewidths=1,
                           s=90)
            ax.scatter(query_data[iv.index(x)], query_data[iv.index(y)], query_data[iv.index(y)], marker="*", s=90)
            plt.show()
        elif x is None or y is None:
            print("Error: x and y axes labels are required when plot mode is enabled.")
        elif x not in iv:
            print(f"Error: column name {x} not found in the training data.")
        elif y not in iv:
            print(f"Error: column name {y} not found in the training data.")
        else:
            pass
    else:
        pass