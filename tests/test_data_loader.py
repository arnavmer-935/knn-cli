import unittest
import os
import tempfile
from knn_cli.data_loader import load_dataset, get_column_names


class TestLoadDataset(unittest.TestCase):

    def _write_temp_csv(self, content):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_loads_correct_number_of_datapoints(self):
        path = self._write_temp_csv("a,b,label\n1.0,2.0,A\n3.0,4.0,B\n5.0,6.0,A\n")
        points, _ = load_dataset(path, "label")
        self.assertEqual(len(points), 3)
        os.unlink(path)

    def test_feature_values_parsed_correctly(self):
        path = self._write_temp_csv("a,b,label\n1.5,2.5,A\n")
        points, _ = load_dataset(path, "label")
        self.assertAlmostEqual(points[0].features[0], 1.5, places=5)
        self.assertAlmostEqual(points[0].features[1], 2.5, places=5)
        os.unlink(path)

    def test_category_parsed_correctly(self):
        path = self._write_temp_csv("a,b,label\n1.0,2.0,Iris-setosa\n")
        points, _ = load_dataset(path, "label")
        self.assertEqual(points[0].category, "Iris-setosa")
        os.unlink(path)

    def test_feature_map_has_correct_keys(self):
        path = self._write_temp_csv("sepal_length,petal_length,label\n1.0,2.0,A\n")
        _, feature_map = load_dataset(path, "label")
        self.assertIn("sepal_length", feature_map)
        self.assertIn("petal_length", feature_map)
        self.assertNotIn("label", feature_map)
        os.unlink(path)

    def test_feature_map_indices_are_sequential(self):
        path = self._write_temp_csv("a,b,c,label\n1.0,2.0,3.0,A\n")
        _, feature_map = load_dataset(path, "label")
        self.assertEqual(feature_map["a"], 0)
        self.assertEqual(feature_map["b"], 1)
        self.assertEqual(feature_map["c"], 2)
        os.unlink(path)

    def test_categorical_column_in_middle(self):
        path = self._write_temp_csv("a,label,b\n1.0,A,2.0\n3.0,B,4.0\n")
        points, feature_map = load_dataset(path, "label")
        self.assertEqual(feature_map["a"], 0)
        self.assertEqual(feature_map["b"], 1)
        self.assertAlmostEqual(points[0].features[0], 1.0, places=5)
        self.assertAlmostEqual(points[0].features[1], 2.0, places=5)
        os.unlink(path)

    def test_categorical_column_first(self):
        path = self._write_temp_csv("label,a,b\nA,1.0,2.0\nB,3.0,4.0\n")
        points, feature_map = load_dataset(path, "label")
        self.assertEqual(feature_map["a"], 0)
        self.assertEqual(feature_map["b"], 1)
        self.assertAlmostEqual(points[0].features[0], 1.0, places=5)
        os.unlink(path)

    def test_features_stored_as_tuple(self):
        path = self._write_temp_csv("a,b,label\n1.0,2.0,A\n")
        points, _ = load_dataset(path, "label")
        self.assertIsInstance(points[0].features, tuple)
        os.unlink(path)

    def test_whitespace_stripped_from_values(self):
        path = self._write_temp_csv("a,b,label\n 1.0 , 2.0 , A \n")
        points, _ = load_dataset(path, "label")
        self.assertAlmostEqual(points[0].features[0], 1.0, places=5)
        self.assertEqual(points[0].category, "A")
        os.unlink(path)

    def test_non_numeric_feature_raises_value_error(self):
        path = self._write_temp_csv("a,b,label\n1.0,notanumber,A\n")
        with self.assertRaises(ValueError):
            load_dataset(path, "label")
        os.unlink(path)

    def test_multiple_categories_loaded(self):
        path = self._write_temp_csv("a,b,label\n1.0,2.0,A\n3.0,4.0,B\n5.0,6.0,C\n")
        points, _ = load_dataset(path, "label")
        categories = {pt.category for pt in points}
        self.assertEqual(categories, {"A", "B", "C"})
        os.unlink(path)


class TestGetColumnNames(unittest.TestCase):

    def _write_temp_csv(self, content):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_returns_all_column_names(self):
        path = self._write_temp_csv("a,b,label\n1.0,2.0,A\n")
        cols = get_column_names(path)
        self.assertEqual(cols, ["a", "b", "label"])
        os.unlink(path)

    def test_strips_whitespace_from_column_names(self):
        path = self._write_temp_csv(" a , b , label \n1.0,2.0,A\n")
        cols = get_column_names(path)
        self.assertEqual(cols, ["a", "b", "label"])
        os.unlink(path)

    def test_returns_correct_count(self):
        path = self._write_temp_csv("a,b,c,d,label\n1.0,2.0,3.0,4.0,A\n")
        cols = get_column_names(path)
        self.assertEqual(len(cols), 5)
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()