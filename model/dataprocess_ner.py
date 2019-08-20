"""
Data process for chiese text classification after raw text process
"""

import os
import json
import logging
import collections
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


# TODO: [x] modify this func for other implementation
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
        features:
            input_ids  : (`List[Int]`) token的id，在chinese模式中就是每个分词的id，对应一个word vector
            input_mask : (`List[Int]`) 真实字符对应1，补全字符对应0
            input_type_ids: (`List[Int]`) 句子标识符，第一句全为0，第二句全为1
    '''
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        # process labels/label_ids
        label_ids = None
        if example.label:
            label_ids = [label_map[label] for label in example.label]
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
            logger.info("*** Example ***")
            logger.info(f"unique_id: {example.unique_id}")
            logger.info("tokens: {}".format(" ".join(tokens)))
            logger.info("input_ids: {}".format(" ".join(
                [str(x) for x in input_ids])))
            logger.info("input_mask: {}".format(" ".join(
                [str(x) for x in input_mask])))
            if label_ids:
                logger.info("input_mask: {}".format(" ".join(
                    [str(x) for x in label_ids])))

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
        examples = []
        for i, (word_seq, label_seq) in enumerate(pairs):
            unique_id = "{}-{}".format(dataset_type, i)
            text_a = word_seq
            label = label_seq.split('\x02')
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, label=label))
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


RawResult = collections.namedtuple("RawResult", ["unique_id", "logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      output_prediction_file, output_nbest_file,
                      output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", [
            "feature_index", "start_index", "end_index", "start_logit",
            "end_logit"
        ])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[
                    0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(
                            start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(feature_index=min_null_feature_index,
                                  start_index=0,
                                  end_index=0,
                                  start_logit=null_start_logit,
                                  end_logit=null_end_logit))
        prelim_predictions = sorted(prelim_predictions,
                                    key=lambda x:
                                    (x.start_logit + x.end_logit),
                                    reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index +
                                                              1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end +
                                                                 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case,
                                            verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(text=final_text,
                                 start_logit=pred.start_logit,
                                 end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(text="",
                                     start_logit=null_start_logit,
                                     end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(
                    0,
                    _NbestPrediction(text="empty",
                                     start_logit=0.0,
                                     end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions
