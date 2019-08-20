from typing import Union, List, Dict
import csv
import os
import codecs
import json
import random
import logging
import argparse
from tqdm import tqdm, trange

from model.dataprocess_ner import *
from model.args import *

from sklearn import metrics
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_transformers import BertTokenizer
from pytorch_transformers import BertForSequenceClassification
from pytorch_transformers.optimization import AdamW
from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):

    processors = {'simpleclasspro': SimpleClassPro, 'stspro': STSPro}

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    task_name = 'stspro'
    processor = processors[task_name]()  # detail
    label_list = processor.get_labels()

    device = torch.device("cuda")
    # state_dict = torch.load(args.state_dict)
    # model = BertForSequenceClassification.from_pretrained(
    #     args.bert_model, state_dict)
    model = BertForSequenceClassification.from_pretrained(args.bert_model)
    model.to(device)

    text_group = args.text.split(',')
    label = int(text_group[-1].strip())
    text_a = text_group[0].strip()
    text_b = text_group[1].strip()

    # test_examples = processor.get_test_examples(args.data_dir)
    test_examples = [InputExample(guid=0, text_a=text_a, text_b=text_b, label=label)]
    test_features = convert_examples_to_features(
        test_examples, label_list, tokenizer)
    all_input_ids = torch.tensor(
        [f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in test_features], dtype=torch.long)
    # all_label_ids = torch.tensor(
    #     [f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids)

    # Run prediction for the test example
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    model.eval()
    for input_ids, input_mask, segment_ids in test_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        # label_ids = label_ids.to(device)

        with torch.no_grad():       ## TODO why use no_grad()?
            logits = model(input_ids, token_type_ids=segment_ids)[0]
            pred = logits.max(1)[1]
            print("*" * 10)
            print("The pred of this sentense is {}".format(label_list[pred]))
            print("*" * 10)


if __name__ == "__main__":
    main(get_test_args())
