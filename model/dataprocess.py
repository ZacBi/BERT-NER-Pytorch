"""
Data process for chiese text classification after raw text process
"""
import csv
import os
import json
import logging
import collections
from typing import Union, List, Dict

import torch

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, unique_id, text_a, text_b=None, label: str = None):
        """
        Constructs a InputExample.
        @params:
            `unique_id`: Unique id for the example.
            `text_a`: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            `text_b`: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            `label`: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask,
                 input_type_ids, label_ids: str):
        """
        @param label_id: the id corresponding to label should be a int, e.g., 
                        label_list = ['O', 'B-PER', 'I-PER'], the label_ids should
                        be [6, 0, 1]
        """
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.label_ids = label_ids


def convert_examples_to_features(examples: List[InputExample],
                                 label_list: List[str],
                                 tokenizer,
                                 max_seq_len: int,
                                 show_example: bool = False):
    '''
    Loads a data file into a list of `InputBatch`s.
    Make some changes for NER (MSRA especially)

    Patch: 
        After trying to predict the cls/label of [PAD], [CLS] and [SEP], 
        I found it is not nercessary to predict them, so for special token mentioned before.
        I label them with 'O'(outside), and caculate CE without regard to them.

    e.g.   max_seq_len: 8\n
    sentence       [CLS]       北       京       市      政      府      [SEP]      [PAD]\n
    input_ids        0         3        4        5      6       7        1          2\n
    labels         [CLS]     B-ORG    I-ORG   I-ORG   I-ORG   I-ORG    [SEP]      [PAD]\n
    label_ids        6         2        3        3      3       3        6          6\n
           
    Args:
        `examples` (`List`):  输入样本，包括text_a, text_b, label, index
        `max_seq_len` (`int`):  文本最大长度(包含[CLS], [SEP]和[PAD])
        `tokenizer` \(`BertTokenizer`\):  分词方法
        `label_list` (`List[str]`): the list of label of each example in examples

    Returns:
        `features`:
            input_ids  : (`List[Int]`) token的id，在chinese模式中就是每个分词的id，对应一个word vector
            input_mask : (`List[Int]`) 真实字符对应1，补全字符对应0
            input_type_ids: (`List[Int]`) 句子标识符，第一句全为0，第二句全为1

    Attention:
        For the reason that tokenizer of pytorhc_transformer will assemble two seperate number, e.g.,
        '10个人' will be ['10', '个', '人'] rather than ['1', '0', '个', '人'], but in MSRA-NER dataset,
        '10个人' will be tagged as ['O', 'O', 'O', 'O]. So i remove tokenizer inner func but retain it 
        in the arguments, for easy modification.
    '''
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        # tokens_a = tokenizer.tokenize(example.text_a)
        tokens_a = example.text_a
        # process labels/label_ids
        label_ids = None
        if example.label:
            label_ids = [label_map[label] for label in example.label]
            assert len(label_ids) == len(tokens_a)
            if len(label_ids) > max_seq_len - 2:
                label_ids = label_ids[:(max_seq_len - 2)]
            label_ids.insert(0, label_map['O'])  # for [CLS]
            label_ids.append(label_map['O'])  # for [SEP]
            while len(label_ids) < max_seq_len:
                label_ids.append(label_map['O'])  # for [PAD]

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_len - 2:
            tokens_a = tokens_a[:(max_seq_len - 2)]

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        input_mask[0], input_mask[-1] = 0, 0

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(input_type_ids) == max_seq_len
        if label_ids:
            assert len(label_ids) == max_seq_len

        if ex_index < 5 and show_example:  # Q: maybe show the first five examples
            pass

        features.append(
            InputFeatures(unique_id=example.unique_id,
                          tokens=tokens,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          input_type_ids=input_type_ids,
                          label_ids=label_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model,
                                  param_model) in zip(named_params_optimizer,
                                                      named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(
                name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer,
                              named_params_model,
                              test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model,
                                  param_model) in zip(named_params_optimizer,
                                                      named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(
                name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(
                    param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        dicts = []
        with open(input_file, 'r', encoding='utf-8') as f_obj:
            infs = json.load(f_obj)
            for inf in infs:
                # inf = inf.strip()
                dicts.append(inf)
        return dicts

    @classmethod
    def _read_txt(cls, input_file, quoterchar=None):
        pass

    @classmethod
    def _read_csv(cls, input_file, quoterchar=None):
        """Read a comma separeted value file."""
        dicts = []
        with open(input_file, 'r', encoding='utf-8') as f_obj:
            for inf in f_obj:
                inf = inf.strip()
                dicts.append(inf)
        return dicts


class MSRAProcessor(DataProcessor):
    '''
        Processor specific for MSRA-NER dataset in ERNIE datasets with format(tsv):
        `厦\x02门\x02与\x02金\x02门\tB-LOC\x02I-LOC\x02O\x02B-LOC\x02I-LOC\n`
        tsv use '\t' to seperate sentence and label list.

        for the reason that in ner we use label seq rather than single label,
        corresponding to a sentence seq
    '''
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]

    def _create_examples(self, pairs, dataset_type):
        """
        Args:
            `pairs`: with pair format (['北', '京'], ['B-ORG', 'I-ORG'])
        """
        examples = []
        for i, (word_seq, label_seq) in enumerate(pairs):
            unique_id = "{}-{}".format(dataset_type, i)
            text_a = word_seq
            label = label_seq
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, label=label))
        return examples

    def _read_tsv(self, input_file: str, quoterchar=None):
        """
            @params:
                input_file str: path to the file
        """
        with open(input_file, 'r', encoding='utf-8') as f_obj:
            reader = csv.reader(f_obj, delimiter="\t", quotechar=quoterchar)
            headers = next(reader)
            pairs = []
            for line in reader:
                pair = (line[0].split('\x02'), line[1].split('\x02'))
                pairs.append(pair)
        return pairs


def read_ner_examples(task_name, input_file, is_training=True):
    """
    input_file should be a data dir rather than a path to tsv/csv file
    """
    processor_dict = {'msra': MSRAProcessor}
    processor = processor_dict[task_name]()
    examples = processor.get_train_examples(os.path.dirname(
        input_file)) if is_training else processor.get_dev_examples(
            os.path.dirname(input_file))
    label_list = processor.get_labels()
    return examples, label_list
