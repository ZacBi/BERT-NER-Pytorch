#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import multiprocessing

from model.dataprocess import convert_examples_to_features

import paddle
import paddle.fluid as fluid

import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

# from model.ernie import ErnieModel


# TODO: a temporary func, annotate it when indeed
def output_correct_example(data_dir):
    pass


def chunk_eval(np_labels, np_infers, np_lens, tag_num, dev_count=1):

    # All the span below, i.e., cur_chunk={'st': idx, 'en', idx + 1, 'type': tag_type},
    # is a left-closed right-open interval.

    def extract_bio_chunk(seq):
        chunks = []
        cur_chunk = None
        # cur_chunk point to the chunk that has been appended to chunks
        null_index = tag_num - 1  # 'O' as last element in label list
        for index in range(len(seq)):
            # idx always point to current char
            tag = seq[index]
            # a brilliant implement to count span without any detail info about the labels/tags
            tag_type = tag // 2
            tag_pos = tag % 2

            if tag == null_index:
                if cur_chunk is not None:
                    chunks.append(cur_chunk)  # cursor for chunk
                    cur_chunk = None
                continue

            if tag_pos == 0:
                if cur_chunk is not None:
                    chunks.append(cur_chunk)
                    cur_chunk = {}
                cur_chunk = {"st": index, "en": index + 1, "type": tag_type}
            else:
                # wrong tag, e.g.
                # TP: O, O, O
                # FP: O, I-ORG, O
                if cur_chunk is None:
                    # st: start, en: end
                    cur_chunk = {
                        "st": index,
                        "en": index + 1,
                        "type": tag_type
                    }
                    continue

                # if the type of tag of succeeding char is the same as cur_chunk,
                # and not be the `B-*`, the succeeding char is regarded as inner part of cur_chunk
                if cur_chunk["type"] == tag_type:
                    cur_chunk["en"] = index + 1
                else:
                    chunks.append(cur_chunk)
                    cur_chunk = {
                        "st": index,
                        "en": index + 1,
                        "type": tag_type
                    }

        if cur_chunk is not None:
            chunks.append(cur_chunk)
        return chunks

    null_index = tag_num - 1
    num_label = 0
    num_infer = 0
    num_correct = 0
    labels = np_labels.reshape([-1]).astype(np.int32).tolist()
    infers = np_infers.reshape([-1]).astype(np.int32).tolist()
    all_lens = np_lens.reshape([dev_count, -1]).astype(np.int32).tolist()

    base_index = 0
    for dev_index in range(dev_count):
        lens = all_lens[dev_index]
        max_len = 0
        for l in lens:
            max_len = max(max_len, l)

        for i in range(len(lens)):
            # + 1 beause we don't need count [CLS]
            # - 2 for the similar reason
            seq_st = base_index + i * max_len + 1
            # TODO: for I label [CLS] and [SEP] as 'O',
            # and don't caculate the loss for them,
            # i.e., the length is caculated by input_mask,
            # but the elem in the position of input_mask
            # where [CLS] and [SEP] place is 0
            # so the seq_en should become seq_st + lens[i]
            seq_en = seq_st + lens[i]
            # seq_en = seq_st + (lens[i] - 2)
            infer_chunks = extract_bio_chunk(infers[seq_st:seq_en])
            label_chunks = extract_bio_chunk(labels[seq_st:seq_en])
            num_infer += len(infer_chunks)
            num_label += len(label_chunks)

            infer_index = 0
            label_index = 0

            while label_index < len(label_chunks) \
                   and infer_index < len(infer_chunks):
                if infer_chunks[infer_index]["st"] \
                    < label_chunks[label_index]["st"]:
                    infer_index += 1
                elif infer_chunks[infer_index]["st"] \
                    > label_chunks[label_index]["st"]:
                    label_index += 1
                else:
                    # chunk match
                    if infer_chunks[infer_index]["en"] \
                        == label_chunks[label_index]["en"] \
                        and infer_chunks[infer_index]["type"] \
                        == label_chunks[label_index]["type"]:
                        num_correct += 1

                    infer_index += 1
                    label_index += 1

        base_index += max_len * len(lens)

    return num_label, num_infer, num_correct


def calculate_f1(num_label, num_infer, num_correct):
    if num_infer == 0:
        precision = 0.0
    else:
        precision = num_correct * 1.0 / num_infer

    if num_label == 0:
        recall = 0.0
    else:
        recall = num_correct * 1.0 / num_label

    if num_correct == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_on_msra(model,
                     processor,
                     args,
                     label_list,
                     tokenizer,
                     device,
                     train_prhase: bool = True,
                     show_example: bool = False):
    if train_prhase:
        eval_examples = processor.get_dev_examples(args.data_dir)
    else:
        eval_examples = processor.get_test_examples(args.data_dir)
    eval_features = convert_examples_to_features(eval_examples,
                                                 label_list,
                                                 tokenizer,
                                                 args.max_seq_length,
                                                 show_example=show_example)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features],
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features],
                                  dtype=torch.long)
    all_input_type_ids = torch.tensor(
        [f.input_type_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in eval_features],
                                 dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_type_ids, all_label_ids,
                              all_input_mask)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)

    label_count = 0
    infer_count = 0
    correct_count = 0
    model.eval()
    for input_ids, input_type_ids, label_ids, input_mask in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        input_type_ids = input_type_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids,
                            token_type_ids=input_type_ids,
                            attention_mask=input_mask,
                            labels=label_ids)
            loss, scores = outputs[:2]
            # np_infers with shape (bath_size:1, max_seq_len)
            np_infers = scores.max(-1)[-1].cpu().numpy()
            np_labels = label_ids.cpu().numpy()
            np_lens = input_mask.sum(-1).cpu().numpy()
            num_label, num_infer, num_correct = chunk_eval(
                np_labels, np_infers, np_lens, args.num_labels)
            if num_correct == np_labels and np_labels == np_infers:
                print(np_labels, '\n', np_infers)
            label_count += num_label
            infer_count += num_infer
            correct_count += num_correct

    return calculate_f1(label_count, infer_count, correct_count)
