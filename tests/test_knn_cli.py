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

class TestCLIValidRuns(unittest.TestCase):

    def setUp(self):
        self.runner = CliRunner()

    def test_basic_prediction_exits_successfully(self):
        result = self.runner.invoke(app, [IRIS, "5", "--p", IRIS_QUERY])
        print(result.output)
        print(result.exception)
        self.assertEqual(result.exit_code, 0)

    def test_prediction_output_is_known_iris_category(self):
        result = self.runner.invoke(app, [IRIS, "5", "--p", IRIS_QUERY])
        self.assertTrue(any(cat in result.output for cat in IRIS_CATEGORIES))

    def test_describe_flag_outputs_column_names(self):
        result = self.runner.invoke(app, [IRIS, "5", "--p", IRIS_QUERY, "--describe"])
        self.assertEqual(result.exit_code, 0)
        for col in IRIS_COLUMNS:
            self.assertIn(col, result.output)

    def test_manhattan_metric_runs_successfully(self):
        result = self.runner.invoke(app, [IRIS, "5", "--p", IRIS_QUERY, "--m", "manh"])
        self.assertEqual(result.exit_code, 0)

    def test_cosine_metric_runs_successfully(self):
        result = self.runner.invoke(app, [IRIS, "5", "--p", IRIS_QUERY, "--m", "cos"])
        print(result.output)
        self.assertEqual(result.exit_code, 0)

    def test_k_equals_1_runs_successfully(self):
        result = self.runner.invoke(app, [IRIS, "1", "--p", IRIS_QUERY])
        self.assertEqual(result.exit_code, 0)

if __name__ == "__main__":
    unittest.main()