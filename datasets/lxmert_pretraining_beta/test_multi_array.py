import nlp
import nlp.features as features
import numpy as np
from nlp.arrow_writer import ArrowWriter

my_features = {
    "text": features.Array2D(dtype="float32"),
    # "image": features.Array2D(dtype="float32"),
    "source": nlp.Value("string")
}


dict_example_0 = {
    # "image": np.random.rand(5, 5).astype("float32"),
    "source": "1",
    "text": np.random.rand(36, 2048).astype("float32"),
}

dict_example_1 = {
    # "image": np.random.rand(5, 5).astype("float32"),
    "text": np.random.rand(36, 2048).astype("float32"),
    "source": "2"
}

my_features = nlp.Features(my_features)
writer = ArrowWriter(data_type=my_features.type, path="/tmp/beta.arrow")
my_examples = [(0, dict_example_0), (1, dict_example_1)]
for key, record in my_examples:
    example = my_features.encode_example(record)
    writer.write(example)
num_examples, num_bytes = writer.finalize()
dataset = nlp.Dataset.from_file("/tmp/arrow_data.arrow")

print(dir(writer))
# throws an error
# print(dataset[0])
# this works
# print(np.vstack(dataset["image"][0].tolist()).shape)
