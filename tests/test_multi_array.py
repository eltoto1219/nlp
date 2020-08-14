import os
import tempfile
from time import time
from unittest import TestCase

import numpy as np

import nlp
from nlp.arrow_writer import ArrowWriter
from nlp.features import Array2D, Features, Value


class MultiDimArrayTest(TestCase):
    def test_write(self):

        my_features = {
            "matrix": Array2D(dtype="float32"),
            "image": Array2D(dtype="float32"),
            "source": Value("string"),
        }

        dict_example_0 = {
            "image": np.random.rand(5, 5).astype("float32"),
            "source": "foo",
            "matrix": np.random.rand(16, 256).astype("float32"),
        }

        dict_example_1 = {
            "image": np.random.rand(5, 5).astype("float32"),
            "matrix": np.random.rand(16, 256).astype("float32"),
            "source": "bar",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:

            my_features = Features(my_features)
            writer = ArrowWriter(features=my_features, path=os.path.join(tmp_dir, "beta.arrow"))
            my_examples = [(0, dict_example_0), (1, dict_example_1)]
            for key, record in my_examples:
                example = my_features.encode_example(record)
                writer.write(example)
            num_examples, num_bytes = writer.finalize()
            dataset = nlp.Dataset.from_file(os.path.join(tmp_dir, "beta.arrow"))

            matrix_column = dataset["matrix"]
            self.assertIsInstance(matrix_column, list)
            self.assertIsInstance(matrix_column[0], list)
            self.assertIsInstance(matrix_column[0][0], list)
            self.assertEqual(np.array(matrix_column).shape, (2, 16, 256))

            matrix_field_of_first_example = dataset[0]["matrix"]
            self.assertIsInstance(matrix_field_of_first_example, list)
            self.assertIsInstance(matrix_field_of_first_example, list)
            self.assertEqual(np.array(matrix_field_of_first_example).shape, (16, 256))

            matrix_field_of_first_two_examples = dataset[:2]["matrix"]
            self.assertIsInstance(matrix_field_of_first_two_examples, list)
            self.assertIsInstance(matrix_field_of_first_two_examples[0], list)
            self.assertIsInstance(matrix_field_of_first_two_examples[0][0], list)
            self.assertEqual(np.array(matrix_field_of_first_two_examples).shape, (2, 16, 256))
            with dataset.formated_as("numpy"):
                self.assertEqual(dataset["matrix"].shape, (2, 16, 256))
                self.assertEqual(dataset[0]["matrix"].shape, (16, 256))
                self.assertEqual(dataset[:2]["matrix"].shape, (2, 16, 256))

    def test_write_batch(self):

        my_features = {
            "matrix": Array2D(dtype="float32"),
            "image": Array2D(dtype="float32"),
            "source": Value("string"),
        }

        dict_examples = {
            "image": np.random.rand(2, 5, 5).astype("float32").tolist(),
            "source": ["foo", "bar"],
            "matrix": np.random.rand(2, 16, 256).astype("float32").tolist(),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:

            my_features = Features(my_features)
            writer = ArrowWriter(features=my_features, path=os.path.join(tmp_dir, "beta.arrow"))
            # dict_examples = my_features.encode_batch(dict_examples)
            writer.write_batch(dict_examples)
            num_examples, num_bytes = writer.finalize()
            dataset = nlp.Dataset.from_file(os.path.join(tmp_dir, "beta.arrow"))

            matrix_column = dataset["matrix"]
            self.assertIsInstance(matrix_column, list)
            self.assertIsInstance(matrix_column[0], list)
            self.assertIsInstance(matrix_column[0][0], list)
            self.assertEqual(np.array(matrix_column).shape, (2, 16, 256))

            matrix_field_of_first_example = dataset[0]["matrix"]
            self.assertIsInstance(matrix_field_of_first_example, list)
            self.assertIsInstance(matrix_field_of_first_example, list)
            self.assertEqual(np.array(matrix_field_of_first_example).shape, (16, 256))

            matrix_field_of_first_two_examples = dataset[:2]["matrix"]
            self.assertIsInstance(matrix_field_of_first_two_examples, list)
            self.assertIsInstance(matrix_field_of_first_two_examples[0], list)
            self.assertIsInstance(matrix_field_of_first_two_examples[0][0], list)
            self.assertEqual(np.array(matrix_field_of_first_two_examples).shape, (2, 16, 256))
            with dataset.formated_as("numpy"):
                self.assertEqual(dataset["matrix"].shape, (2, 16, 256))
                self.assertEqual(dataset[0]["matrix"].shape, (16, 256))
                self.assertEqual(dataset[:2]["matrix"].shape, (2, 16, 256))
