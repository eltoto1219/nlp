"""TODO(art): Add a description here."""

from __future__ import absolute_import, division, print_function

import json
import os

import nlp


# TODO(art): BibTeX citation
_CITATION = """\
@InProceedings{anli,
  author = {Chandra, Bhagavatula and Ronan, Le Bras and Chaitanya, Malaviya and Keisuke, Sakaguchi and Ari, Holtzman
    and Hannah, Rashkin and Doug, Downey and Scott, Wen-tau Yih and Yejin, Choi},
  title = {Abductive Commonsense Reasoning},
  year = {2020}
}"""

# TODO(art):
_DESCRIPTION = """\
the Abductive Natural Language Inference Dataset from AI2
"""
_DATA_URL = "https://storage.googleapis.com/ai2-mosaic/public/alphanli/alphanli-train-dev.zip"


class ArtConfig(nlp.BuilderConfig):
    """BuilderConfig for Art."""

    def __init__(self, **kwargs):
        """BuilderConfig for Art.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ArtConfig, self).__init__(
            version=nlp.Version("0.1.0", "New split API (https://tensorflow.org/datasets/splits)"), **kwargs
        )


class Art(nlp.GeneratorBasedBuilder):
    """TODO(art): Short description of my dataset."""

    # TODO(art): Set up version.
    VERSION = nlp.Version("0.1.0")
    BUILDER_CONFIGS = [
        ArtConfig(
            name="anli",
            description="""\
          the Abductive Natural Language Inference Dataset from AI2.
          """,
        ),
    ]

    def _info(self):
        # TODO(art): Specifies the nlp.DatasetInfo object
        return nlp.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # nlp.features.FeatureConnectors
            features=nlp.Features(
                {
                    "observation_1": nlp.Value("string"),
                    "observation_2": nlp.Value("string"),
                    "hypothesis_1": nlp.Value("string"),
                    "hypothesis_2": nlp.Value("string"),
                    "label": nlp.features.ClassLabel(num_classes=3)
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://leaderboard.allenai.org/anli/submissions/get-started",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(art): Downloads the data and defines the splits
        # dl_manager is a nlp.download.DownloadManager that can be used to
        # download and extract URLs
        dl_dir = dl_manager.download_and_extract(_DATA_URL)
        return [
            nlp.SplitGenerator(
                name=nlp.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(dl_dir, "dev.jsonl"),
                    "labelpath": os.path.join(dl_dir, "dev-labels.lst"),
                },
            ),
            nlp.SplitGenerator(
                name=nlp.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(dl_dir, "train.jsonl"),
                    "labelpath": os.path.join(dl_dir, "train-labels.lst"),
                },
            ),
        ]

    def _generate_examples(self, filepath, labelpath):
        """Yields examples."""
        # TODO(art): Yields (key, example) tuples from the dataset
        data = []
        for line in open(filepath, encoding="utf-8"):
            data.append(json.loads(line))
        labels = []
        with open(labelpath, encoding="utf-8") as f:
            for word in f:
                labels.append(word)
        for idx, row in enumerate(data):
            yield idx, {
                "observation_1": row["obs1"],
                "observation_2": row["obs2"],
                "hypothesis_1": row["hyp1"],
                "hypothesis_2": row["hyp2"],
                "label": labels[idx],
            }
