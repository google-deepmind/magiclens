# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=redefined-outer-name,missing-module-docstring,g-importing-member,missing-function-docstring,g-bare-generic,g-doc-args,missing-class-docstring
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
from typing import Any, List, Union
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image


@dataclass
class QueryExample:
    qid: str
    qtokens: np.ndarray
    qimage: np.ndarray
    target_iid: Union[int, str, List[int], List[str], None] # can be int or 
    retrieved_iids: List[Union[int, str]] # ranked by score, can be str (cirr) or int (circo)
    retrieved_scores: List[float] # ranked by order


@dataclass
class IndexExample:
    iid: Union[int, str]
    iimage: np.ndarray
    itokens: np.ndarray


@dataclass
class Dataset:
    name: str
    query_examples: List[QueryExample] = field(default_factory=list)
    k_range: List[int] = field(default_factory=lambda: [10, 50])
    # write_to_file_header: Dict[str, Any] = field(default_factory=dict)
    index_examples: List[IndexExample] = field(default_factory=list)

    def evaluate_recall(self):
        ret_dict = {k: [] for k in self.k_range}

        for q_example in self.query_examples:
            assert len(q_example.retrieved_iids) > 0, "retrieved_iids is empty"
            for k in self.k_range:
                recalled = False
                if isinstance(q_example.target_iid, list):
                    for one_target_iid in q_example.target_iid:
                        if one_target_iid in q_example.retrieved_iids[:k]:
                            recalled = True
                elif isinstance(q_example.target_iid, int) or isinstance(q_example.target_iid, str):
                    if q_example.target_iid in q_example.retrieved_iids[:k]:
                        recalled = True
                else:
                    raise ValueError(f"target_iid is of type {type(q_example.target_iid)}")

                if recalled:
                    ret_dict[k].append(1)
                else:
                    ret_dict[k].append(0)
        # calculation
        total_ex = len(self.query_examples)
        ret_dict = {k: (sum(v) / total_ex) * 100 for k, v in ret_dict.items()}
        print("Recalls: ", ret_dict)

        return ret_dict


    def write_to_file(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dict_to_write = dict()
        for q_example in self.query_examples:
            dict_to_write[q_example.qid] = q_example.retrieved_iids[:50]
        output_file = os.path.join(output_dir, f"{self.name}_results.json")
        with open(output_file, "w") as f:
            json.dump(dict_to_write, f, indent=4)
        print("Results are written to file", output_file)


def process_img(image_path: str, size: int) -> np.ndarray:
    """Process a single image to 224x224 and normalize."""
    img = Image.open(image_path).convert("RGB")
    ima = jnp.array(img)[jnp.newaxis, ...] # [1, 224, 224, 3]
    ima = ima / (ima.max() + 1e-12)  # avoid 0 division
    ima = jax.image.resize(ima, (1, size, size, 3), method='bilinear')
    return np.array(ima)


def build_fiq_dataset(dataset_name: str, tokenizer: Any) -> Dataset:
    eval_dataset = Dataset(dataset_name)
    subtask = dataset_name.split("-")[1]
    queries = json.load(open(f"./data/fiq/captions/cap.{subtask}.val.json"))
    index_img_ids = json.load(open(f"./data/fiq/image_splits/split.{subtask}.val.json"))
    index_image_folder = "./data/fiq/images"

    null_tokens = tokenizer("")  # used for index example
    null_tokens = np.array(null_tokens)

    def process_index_example(index_img_id):
        img_path = os.path.join(index_image_folder, index_img_id + ".png")
        ima = process_img(img_path, 224)
        return IndexExample(iid=index_img_id, iimage=ima, itokens=null_tokens)

    def process_query_example(query):
        qid = query['candidate']
        qtext = " and ".join(query['captions'])
        qimage_path = os.path.join(index_image_folder, query['candidate'] + ".png")
        ima = process_img(qimage_path, 224)
        qtokens = tokenizer(qtext)
        return QueryExample(qid=qid, qtokens=qtokens, qimage=ima, target_iid=query['target'], retrieved_iids=[], retrieved_scores=[])

    with ThreadPoolExecutor() as executor:
        print("Preparing index examples...")
        index_example_futures = {executor.submit(process_index_example, index_img_id): index_img_id for index_img_id in index_img_ids}

        with tqdm(total=len(index_img_ids), desc="Index examples") as progress:
            for future in as_completed(index_example_futures):
                index_example = future.result()
                eval_dataset.index_examples.append(index_example)
                progress.update(1)

        print("Prepared index examples.")

        print("Preparing query examples...")
        query_futures = {executor.submit(process_query_example, query): query for query in queries}

        with tqdm(total=len(queries), desc="Query examples") as progress:
            for future in as_completed(query_futures):
                q_example = future.result()
                eval_dataset.query_examples.append(q_example)
                progress.update(1)

    return eval_dataset


def build_circo_dataset(dataset_name: str, tokenizer: Any) -> Dataset:
    eval_dataset = Dataset(dataset_name)
    queries = json.load(open("./data/circo/annotations/test.json"))
    coco_info = json.load(open("./data/circo/COCO2017_unlabeled/annotations/image_info_unlabeled2017.json"))
    index_img_ids = [img_info['id'] for img_info in coco_info['images']]
    index_image_folder = "./data/circo/COCO2017_unlabeled/unlabeled2017"

    def image_id2name(image_id):
        return str(image_id).zfill(12) + '.jpg'

    null_tokens = tokenizer("")  # used for index example
    null_tokens = np.array(null_tokens)

    def process_index_example(index_img_id):
        img_path = os.path.join(index_image_folder, image_id2name(index_img_id))
        ima = process_img(img_path, 224)
        return IndexExample(iid=index_img_id, iimage=ima, itokens=null_tokens)

    def process_query_example(query):
        qid = query['id']
        qtext = f"find {query['shared_concept']} but {query['relative_caption']}"
        qimage_path = os.path.join(index_image_folder, image_id2name(query['reference_img_id']))
        ima = process_img(qimage_path, 224)
        qtokens = np.array(tokenizer(qtext))
        # circo test does not provide target id.
        return QueryExample(qid=qid, qtokens=qtokens, qimage=ima, target_iid=0, retrieved_iids=[], retrieved_scores=[])

    with ThreadPoolExecutor() as executor:
        print("Preparing index examples...")
        index_example_futures = {executor.submit(process_index_example, index_img_id): index_img_id for index_img_id in index_img_ids}

        with tqdm(total=len(index_img_ids), desc="Index examples") as progress:
            for future in as_completed(index_example_futures):
                index_example = future.result()
                eval_dataset.index_examples.append(index_example)
                progress.update(1)

        print("Prepared index examples.")

        print("Preparing query examples...")
        query_futures = {executor.submit(process_query_example, query): query for query in queries}

        with tqdm(total=len(queries), desc="Query examples") as progress:
            for future in as_completed(query_futures):
                q_example = future.result()
                eval_dataset.query_examples.append(q_example)
                progress.update(1)

    return eval_dataset


# TODO test
def build_dtin_dataset(dataset_name: str, tokenizer: Any) -> Dataset:
    eval_dataset = Dataset(dataset_name)
    all_domains = ['cartoon', 'origami', 'toy', 'sculpture']   # only evaluate on one domain per run
    target_domain = dataset_name.split("-")[1]
    query_entries = open("./data/dtin/imgnet_real_query.txt").readlines()
    index_entries = open("./data/dtin/imgnet_targets.txt").readlines()

    # debug
    # query_entries = query_entries[:10]
    # index_entries = index_entries[:10]
    null_tokens = tokenizer("")  # used for index example
    null_tokens = np.array(null_tokens)

    query_text_template = "find this object in {}"
    # parse queries and targets

    def process_index_example(index_entry):
        iimage_path, iid = index_entry.split()
        iimage_path = "/".join(iimage_path.split("/")[1:])  # remove the first 'imgnet/' as we are using dtin now
        iimage_path = os.path.join("./data/dtin/", iimage_path)

        ima = process_img(iimage_path, 224)
        return IndexExample(iid=int(iid), iimage=ima, itokens=null_tokens)

    def process_query(query_entry, domain_id, domain):
        qimg_path, class_id = query_entry.split()
        qimg_path = "/".join(qimg_path.split("/")[1:])  # remove the first 'imgnet/' as we are using dtin now
        qimage_path = os.path.join("./data/dtin/", qimg_path)
        target_iid = domain_id * 1000 + int(class_id)
        qid = qimg_path.split("/")[-1]

        qtext = query_text_template.format(domain)
        qimage = process_img(qimage_path, 224)
        qtokens = np.array(tokenizer(qtext))
        return QueryExample(qid=qid + "-class-" + class_id + "-to-" + domain, qtokens=qtokens, qimage=qimage, target_iid=target_iid, retrieved_iids=[], retrieved_scores=[])

    
    domain_id = all_domains.index(target_domain)
    with ThreadPoolExecutor() as executor:
        print("Preparing query examples...")
        query_futures = {executor.submit(process_query, query_entry, domain_id, target_domain): (query_entry, domain_id, target_domain) for query_entry in query_entries}

        with tqdm(total=len(query_entries), desc="Query examples") as progress:
            for future in as_completed(query_futures):
                q_example = future.result()
                eval_dataset.query_examples.append(q_example)
                progress.update(1)

    print("Prepared query examples.")

    print("Preparing target examples...")

    with ThreadPoolExecutor() as executor:
        print("Preparing index examples...")
        index_example_futures = {executor.submit(process_index_example, index_entry): index_entry for index_entry in index_entries}

        with tqdm(total=len(index_entries), desc="Index examples") as progress:
            for future in as_completed(index_example_futures):
                index_example = future.result()
                eval_dataset.index_examples.append(index_example)
                progress.update(1)

    # import pdb; pdb.set_trace()
    print("Prepared index examples.")
    return eval_dataset
