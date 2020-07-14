# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace NLP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
# """
# TODO: first convert dataset to arrow format
# """
#
# """LXMERT multimodal pretraining dataset beta"""
from __future__ import absolute_import, division, print_function
import hashlib
import logging
import nlp

_DESCRIPTION = """\
LXMERT multimodal pretraining dataset beta
"""

_CITATION = """\
@inproceedings{tan2019lxmert,
  title={LXMERT: Learning Cross-Modality Encoder Representations from Transformers},
  author={Tan, Hao and Bansal, Mohit},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year={2019}
}
"""

_DL_URLS = {
    "mscoco_train_imgs": "https://nlp1.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip",
    "mscoco_val_imgs": "https://nlp1.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip",
    "vg_trainval_imgs": "https://nlp1.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/vg_gqa_obj36.zip",
    "vg_testdev_imgs": "https://nlp1.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/gqa_testdev_obj36.zip",
    "mscoco_train_txt": "https://nlp1.cs.unc.edu/data/lxmert_data/lxmert/mscoco_train.json",
    "mscoco_val_txt": "https://nlp1.cs.unc.edu/data/lxmert_data/lxmert/mscoco_nominival.json",
    "vg_trainval_txt": "https://nlp1.cs.unc.edu/data/lxmert_data/lxmert/vgnococo.json",
    "mscoco_minival_txt": "https://nlp1.cs.unc.edu/data/lxmert_data/lxmert/mscoco_minival.json",
    "ans": "https://raw.githubusercontent.com/airsplay/lxmert/master/data/lxmert/all_ans.json",
}


_DEFAULT_VERSION = nlp.Version("0.0.1", "beta test for multi-modal.")


class LxmertPretrainingBetaConfig(nlp.BuilderConfig):
    """BuilderConfig for LXMERT Pretraining Beta"""

    def __init__(self, **kwargs):
        """
        Args: **kwargs: keyword arguments forwarded to super.
        """
        super(LxmertPretrainingBetaConfig, self).__init__(**kwargs)


def _get_url_hashes(path):
    """Get hashes of urls in file."""
    urls = _read_text_file(path)

    def url_hash(u):
        h = hashlib.sha1()
        try:
            u = u.encode("utf-8")
        except UnicodeDecodeError:
            logging.error("Cannot hash url: %s", u)
        h.update(u)
        return h.hexdigest()

    return {url_hash(u): True for u in urls}


def _read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


class LxmertPretrainingBeta(nlp.GeneratorBasedBuilder):
    """CNN/DailyMail non-anonymized summarization dataset."""

    BUILDER_CONFIGS = [
        LxmertPretrainingBetaConfig(
            name="lxmert-full-pretrain",
            description="GQA, VQA, MSCOCO, VG train + val data",
            version=_DEFAULT_VERSION)
    ]

    def _info(self):
        # TODO
        pass

    """
    def _vocab_text_gen(self, paths):
        for _, ex in self._generate_examples(paths):
            yield " ".join([ex[_ARTICLE], ex[_HIGHLIGHTS]])

    def _split_generators(self, dl_manager):
        dl_paths = dl_manager.download_and_extract(_DL_URLS)
        train_files = _subset_filenames(dl_paths, nlp.Split.TRAIN)
        # Generate shared vocabulary

        return [
            nlp.SplitGenerator(name=nlp.Split.TRAIN, gen_kwargs={"files": train_files}),
            nlp.SplitGenerator(
                name=nlp.Split.VALIDATION, gen_kwargs={"files": _subset_filenames(dl_paths, nlp.Split.VALIDATION)}
            ),
            nlp.SplitGenerator(name=nlp.Split.TEST, gen_kwargs={"files": _subset_filenames(dl_paths, nlp.Split.TEST)}),
        ]

    def _generate_examples(self, files):
        for p in files:
            article, highlights = _get_art_abs(p, self.config.version)
            if not article or not highlights:
                continue
            fname = os.path.basename(p)
            yield fname, {_ARTICLE: article, _HIGHLIGHTS: highlights}
    """


if __name__ == "__main__":
    dataset = LxmertPretrainingBeta(name="3.0.0", splits=["train"])
    print([d for d in dir(dataset) if "__" not in d and d[0] != "_"])
    print(dataset.config)
    print(dataset.builder_configs)
    print(dataset.manual_download_instructions)
    print(dataset.info)
