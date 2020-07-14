from __future__ import absolute_import, division, print_function
import subprocess
import os
import sys
import csv
import base64
import time
import json
from tqdm import tqdm

import nlp
import nlp.features as features
import numpy as np
from nlp.arrow_writer import ArrowWriter


csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

# I will only do for minival for now for testing purposes
# download the following links manually and place intp the playground dir
_DL_URLS = {
    # "val_img": "https://nlp1.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip",
    "all_ans": "https://raw.githubusercontent.com/airsplay/lxmert/master/data/lxmert/all_ans.json",
    "minival": "https://nlp1.cs.unc.edu/data/lxmert_data/vqa/minival.json"
}


class AnswerTable:
    ANS_CONVERT = {
        "a man": "man",
        "the man": "man",
        "a woman": "woman",
        "the woman": "woman",
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
        'grey': 'gray',
    }

    def __init__(self, dsets=None):
        self.all_ans = json.load(open("playground/all_ans.json"))
        if dsets is not None:
            dsets = set(dsets)
            # If the answer is used in the dsets
            self.anss = [ans['ans'] for ans in self.all_ans if
                         len(set(ans['dsets']) & dsets) > 0]
        else:
            self.anss = [ans['ans'] for ans in self.all_ans]
        self.ans_set = set(self.anss)

        self._id2ans_map = self.anss
        self._ans2id_map = {ans: ans_id for ans_id, ans in enumerate(self.anss)}

        assert len(self._id2ans_map) == len(self._ans2id_map)
        for ans_id, ans in enumerate(self._id2ans_map):
            assert self._ans2id_map[ans] == ans_id

    def convert_ans(self, ans):
        if len(ans) == 0:
            return ""
        ans = ans.lower()
        if ans[-1] == '.':
            ans = ans[:-1].strip()
        if ans.startswith("a "):
            ans = ans[2:].strip()
        if ans.startswith("an "):
            ans = ans[3:].strip()
        if ans.startswith("the "):
            ans = ans[4:].strip()
        if ans in self.ANS_CONVERT:
            ans = self.ANS_CONVERT[ans]
        return ans

    def ans2id(self, ans):
        return self._ans2id_map[ans]

    def id2ans(self, ans_id):
        return self._id2ans_map[ans_id]

    def ans2id_map(self):
        return self._ans2id_map.copy()

    def id2ans_map(self):
        return self._id2ans_map.copy()

    def used(self, ans):
        return ans in self.ans_set

    def all_answers(self):
        return self.anss.copy()

    @property
    def num_answers(self):
        return len(self.anss)


def load_obj_tsv(fname, topk=300):
    data = {}
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):
            img_id = item.pop("img_id")

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])

            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data[img_id] = item
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data


processes = []
to_unzip = []
# Download data
print("download data")
for k, url in tqdm(_DL_URLS.items()):
    ftype = url.split("/")[-1]
    fname = "playground/" + ftype
    if fname[-3:] == "zip":
        to_unzip.append(fname)
    if os.path.isfile(fname) or fname[:-3] not in os.listdirs("playground/"):
        continue
    else:
        os.mknod(fname)

    command = f'wget --no-check-certificate {url} -O {fname}'

    processes.append(
        subprocess.run(
            command,
            shell=True,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
        )
    )

assert all(list(map(lambda x: isinstance(x, subprocess.CompletedProcess), processes)))

# unzip data
total_time = 0
for fname in to_unzip:
    command = f"unzip  {fname} -D {fname[:-3]} && rm {fname}"
    process = subprocess.run(
        command,
        shell=True,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
    )
    while not isinstance(process, subprocess.CompletedProcess):
        time.sleep(1)
        total_time += 1
        print(f"unzipping, time: {total_time} s", end="\r", flush=True)

with open("playground/minival.json", "r") as f:
    items = json.load(f)
images = load_obj_tsv("playground/mscoco_imgfeat/val2014_obj36.tsv")
answer_table = AnswerTable()

# pre-pre process
new = {}
raw_labels = []
for i in items:
    img = i.pop("img_id")
    image_data = images.get(img)
    entry = image_data
    if image_data is None:
        continue
    i.pop("answer_type")
    i.pop("question_type")
    i.pop("question_id")
    entry = {**entry, **i}
    sent = entry["sent"]
    labels = entry["label"]
    label = list(labels.keys())
    label_conf = list(labels.values())
    for l in label:
        raw_labels.append(answer_table.convert_ans(l))
    entry["label"] = [answer_table.ans2id(answer_table.convert_ans(x)) for x in label]
    entry["label_conf"] = label_conf
    question = sent if "?" in sent else None
    sent = sent if "?" not in sent else None
    entry["question"] = question
    entry["sent"] = sent
    entry["img_id"] = img
    for k in entry:
        if k not in image_data and k != "img_id":
            entry[k] = [entry[k]]
    if img not in new:
        new[img] = entry
    else:
        cur = new[img]
        assert len(cur) == len(entry), f'{len(cur)}, {len(entry)}'
        for k in cur:
            if k not in image_data and k != "img_id":
                temp = entry[k] + cur[k]
                new_v = [x for x in temp if x is not None]
                if not new_v:
                    new_v = ["<NONE>"]
                cur[k] = new_v

        new[img] = cur

new = list(new.values())


"""
labels that need to be converted to pyarrow features
"""

# need to make sorting better

my_features = {
    "image": features.Array2D(dtype="float32"),
    "img_id": nlp.Value("string"),
    "boxes": nlp.features.Array2D(dtype="int32"),
    "img_h": nlp.Value("int32"),
    "img_w": nlp.Value("int32"),
    "labels": nlp.features.Array2D(dtype="int32"),
    "labels_confidence": nlp.features.Array2D(dtype="float32"),
    "num_boxes": nlp.Value("int32"),
    "attrs_id": nlp.features.Sequence(nlp.ClassLabel(num_classes=400)),
    "objs_id": nlp.features.Sequence(nlp.ClassLabel(num_classes=1600)),
    "attrs_confidence": nlp.features.Sequence(nlp.Value("float32")),
    "objs_confidence": nlp.features.Sequence(nlp.Value("float32")),
    "captions": nlp.features.Sequence(nlp.Value("string")),
    "questions": nlp.features.Sequence(nlp.Value("string")),
}


ex = {
    "image": new[0]["features"].astype("float32"),
    "img_id": str(new[0]["img_id"]),
    "boxes": new[0]["boxes"],
    "img_h": new[0]["img_h"],
    "img_w": new[0]["img_w"],
    "labels": new[0]["label"],
    "labels_confidence": new[0]["label_conf"],
    "num_boxes": new[0]["num_boxes"],
    "attrs_id": new[0]["attrs_id"],
    "objs_id": new[0]["objects_id"],
    "attrs_confidence": new[0]["attrs_conf"],
    "objs_confidence": new[0]["objects_conf"],
    "captions": new[0]["sent"],
    "questions": new[0]["question"],
}


my_features = nlp.Features(my_features)
writer = ArrowWriter(data_type=my_features.type, path="/tmp/beta.arrow")
my_examples = [(0, ex), ]
for key, record in my_examples:
    example = my_features.encode_example(record)
    writer.write(example)
num_examples, num_bytes = writer.finalize()
dataset = nlp.Dataset.from_file("/tmp/beta.arrow")
print(dataset)
