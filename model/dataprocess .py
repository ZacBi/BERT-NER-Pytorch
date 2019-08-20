"""
Data process for chiese text classification after raw text process
"""

import os
import json
import logging
from typing import Union, List, Dict

import torch

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, unique_id, text_a, text_b=None, label: str = None):
        """Constructs a InputExample.
        Args:
            unique_id: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
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
                        be [0, 1, 2]
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
                                 seq_length: int,
                                 show_example: bool = False):
    '''Loads a data file into a list of `InputBatch`s.
    Make some changes for NER (MSRA especially)

    Args:
        examples: [List] 输入样本，包括question, label, index
        seq_length: [int] 文本最大长度
        tokenizer: [Method] 分词方法
        label_list List[str]: the list of label of each example in examples

    Returns:
        features:
            input_ids  : [ListOfInt] token的id，在chinese模式中就是每个分词的id，对应一个word vector
            input_mask : [ListOfInt] 真实字符对应1，补全字符对应0
            input_type_ids: [ListOfInt] 句子标识符，第一句全为0，第二句全为1
    '''
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                # TODO: wdnmd, it's really to truncate it?
                tokens_a = tokens_a[:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        label_ids = None
        if example.label:
            label_ids = [label_map[label] for label in example.label]

        if ex_index < 5 and show_example:  # Q: maybe show the first five examples
            logger.info("*** Example ***")
            logger.info(f"unique_id: {example.unique_id}")
            logger.info("tokens: {}".format(" ".join([tokens])))
            logger.info("input_ids: {}".format(" ".join(
                [str(x) for x in input_ids])))
            logger.info("input_mask: {}".format(" ".join(
                [str(x) for x in input_mask])))
            # logger.info("input_type_ids: {}".format(" ".join(
            #     [str(x) for x in input_type_ids])))

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
        return [
            "CLS", "SEP", "B-PER", "I-PER", "B-ORG",
            "I-ORG", "B-LOC", "I-LOC", "O"
        ]

    def _create_examples(self, pairs, dataset_type):
        examples = []
        for i, (word_seq, label_seq) in enumerate(pairs):
            unique_id = "{}-{}".format(dataset_type, i)
            text_a = word_seq
            label = label_seq.split('\x02')
            examples.append(
                InputExample(unique_id=unique_id,
                             text_a=text_a,
                             label=label))
        return examples

    def _read_tsv(self, input_file: str, quoterchar=None):
        """
            @params:
                input_file str: path to the file
        """
        pairs = []
        with open(input_file, 'r', encoding='utf-8') as f_obj:
            for line in f_obj.readlines()[1:]:
                word_seq, label_seq = line.strip().split('\t')
                pairs.append((word_seq, label_seq))
        return pairs
