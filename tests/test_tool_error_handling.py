import unittest
import typer
from typer.testing import CliRunner
from knn_cli.cli import main

app = typer.Typer()
app.command()(main)

IRIS = "data/iris.data"
IRIS_QUERY = "5.1 3.5 1.4 0.2"
IRIS_CATEGORIES = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}
IRIS_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

class TestCLIErrorCases(unittest.TestCase):

    def setUp(self):
        self.runner = CliRunner()

    def test_missing_query_point_exits_with_error(self):
        result = self.runner.invoke(app, [IRIS, "5"])
        self.assertEqual(result.exit_code, 1)

    def test_nonexistent_dataset_exits_with_error(self):
        result = self.runner.invoke(app, ["data/fake.data", "5", "--p", IRIS_QUERY])
        self.assertEqual(result.exit_code, 1)

    def test_k_zero_exits_with_error(self):
        result = self.runner.invoke(app, [IRIS, "0", "--p", IRIS_QUERY])
        self.assertEqual(result.exit_code, 1)

    def test_k_exceeds_dataset_size_exits_with_error(self):
        result = self.runner.invoke(app, [IRIS, "9999", "--p", IRIS_QUERY])
        self.assertEqual(result.exit_code, 1)

    def test_query_dimension_mismatch_exits_with_error(self):
        result = self.runner.invoke(app, [IRIS, "5", "--p", "1.0 2.0"])
        self.assertEqual(result.exit_code, 1)

    def test_non_numeric_query_value_exits_with_error(self):
        result = self.runner.invoke(app, [IRIS, "5", "--p", "1.0 abc 3.0 4.0"])
        self.assertEqual(result.exit_code, 1)

    def test_axis_without_plot_flag_exits_with_error(self):
        result = self.runner.invoke(app, [IRIS, "5", "--p", IRIS_QUERY, "--x", "sepal_length"])
        self.assertEqual(result.exit_code, 1)

    def test_z_without_y_exits_with_error(self):
        result = self.runner.invoke(
            app, [IRIS, "5", "--p", IRIS_QUERY, "--plot", "--x", "sepal_length", "--z", "petal_length"]
        )
        self.assertEqual(result.exit_code, 1)

    def test_invalid_axis_feature_name_exits_with_error(self):
        result = self.runner.invoke(
            app, [IRIS, "5", "--p", IRIS_QUERY, "--plot", "--x", "nonexistent_col"]
        )
        self.assertEqual(result.exit_code, 1)

if __name__ == "__main__":
    unittest.main()