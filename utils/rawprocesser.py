"""
Raw data processing includes get standard format text, balance sampling.
"""

import json
import os
import re
import argparse
from random import shuffle, uniform, randint
from typing import Union, List, Dict


def domain_count(input_path, domain_list=None) -> List[str]:
    """
    count num of domain examples 
    """
    f_obj = open(input_path, mode='r', encoding='utf-8')
    domains = dict()
    if domain_list is not None:
        domains = {k: 0 for k in domain_list}
    for line in f_obj:
        label = line.strip().split()[0]
        content = line.strip().split()[-1]
        if domain_list is not None:
            if label in domain_list:
                domains[label] += 1
        else:
            if label not in list(domains.keys()):
                domains[label] = 0
            domains[label] += 1

    return domains


def split(input_path, output_dir):
    f_obj = open(input_path, mode='r', encoding='utf-8')
    for i, line in enumerate(f_obj.readlines()):
        if i % 20 == 0:
            tmp_obj = open(os.path.join(output_dir, f'sample_{i // 20}.txt'),
                           mode='w',
                           encoding='utf-8')
        tmp_obj.write(line)


def handle_json(data_dir: str) -> None:
    """
    Handle txt files especially under SMP2018-ECDT
    """
    expr = r'\w+.json'
    pattern = re.compile(expr)

    combined_data = open(os.path.join(data_dir, 'combined_data.txt'),
                         'w',
                         encoding='utf-8')
    for file_path in os.listdir(data_dir):
        group = pattern.match(file_path)
        if group:
            items = json.load(
                open(os.path.join(data_dir, file_path), 'r', encoding='utf-8'))
            for order, item in items.items():
                query = item['query']
                label = item['label']
                combined_data.writelines(' '.join([label, query, '\n']))

    combined_data.close()


def json_to_texts(input_dir):
    """
    result will be output to the same dir
    """
    files = os.listdir(input_dir)
    types = ['dev', 'train']
    for _file in files:
        suffix = _file.split('.')[-1]
        _type = _file.split('.')[0]
        if suffix != 'json' and _type not in types:
            continue

        f_obj = open(os.path.join(input_dir, _file),
                     mode='r',
                     encoding='utf-8')
        dataset_folder = os.path.join(input_dir, _type)
        os.mkdir(dataset_folder)
        target = dict()
        f_json = json.load(f_obj)
        for idx in f_json:
            item = f_json[idx]
            label = item['label']
            query = item['query']
            if label not in target:
                target[label] = open(os.path.join(dataset_folder,
                                                  f'{_type}_{label}_2018.txt'),
                                     mode='w',
                                     encoding='utf-8')
            target[label].write(f'{query}\n')


def handle_txt(data_dir: str) -> None:
    """
    Handle txt files especially under SMP2017 
    """
    expr = r'\w+\_(\w+)\.\w+'
    # domain_needed = {'cookbook', 'train', 'flight'}
    pattern = re.compile(expr)

    combined_data = open(os.path.join(data_dir, 'combined_data.txt'),
                         'w',
                         encoding='utf-8')
    for txt_path in os.listdir(data_dir):
        group = pattern.match(txt_path)
        label = ''
        if group:
            txt_obj = open(os.path.join(data_dir, txt_path),
                           'r',
                           encoding='utf-8')
            label = group[1]
            for line in txt_obj.readlines():
                combined_data.write(' '.join([label, line]))

    combined_data.close()


def convert_to_standard_text(input_path: str) -> None:
    """
    Conver the informal file to standard text file with format |table|whitespace|sentence|:
    weather 今天东莞天气如何
    """
    frist_file = os.listdir(input_path)[0]
    if frist_file.endswith('txt'):
        handle_txt(input_path)
    else:
        handle_json(input_path)


def sample(input_path: str, output_path: str,
           pos_list: List[str] = None) -> None:
    """
    Extract specific numbers of samples

    @param out_path: out_path should be a dir path rather than file(text, etc.) path
    @param num_samples: get the constent number of each types of sample, except 'others'
    @param pos_list: list the positive types of smaples, e.g., ['weather', 'map', etc.].
                     the sample for whose label not in list will be labeled with 'others'
    """
    i_obj = open(input_path, mode='r', encoding='utf-8')
    o_obj = open(output_path, mode='w', encoding='utf-8')
    damain_static = domain_count(input_path)
    # i_obj = open(input_path, mode='r', encoding='utf-8')
    pos_toatal = 0
    neg_count = 0
    pos_count = 0
    lines = i_obj.readlines()
    shuffle(lines)
    # pos_list = ['map', 'flight', 'weather', 'datetime', 'train']
    if pos_list:
        # pos_list = json.loads(pos_list)
        for label in pos_list:
            pos_toatal += damain_static[label]

        for line in lines:
            label = line.strip().split()[0]
            content = line.strip().split()[-1]
            if label not in pos_list:
                if neg_count > pos_toatal * 1.5:
                    continue
                neg_count += 1
            else:
                pos_count += 1

            o_obj.write(content + '\n')


def convert_text_to_json(input_path: str, output_path: str) -> None:
    input_obj = open(input_path, 'r', encoding='utf-8')
    output_obj = open(output_path, 'w', encoding='utf-8')

    dicts = []

    for line in input_obj.readlines():
        sample = dict()
        sample['label'] = line.split()[0].strip()
        sample['content'] = line.split()[-1].strip()
        dicts.append(sample)

    output_obj.write(json.dumps(dicts))


def de_depulicate(input_path, output_path):
    f_obj = open(input_path, mode='r', encoding='utf-8')
    _unique = set()
    for line in f_obj:
        example = line.strip()
        if example not in _unique:
            _unique.add(example + '\n')

    f_de_obj = open(output_path, mode='w', encoding='utf-8')
    f_de_obj.writelines(list(_unique))


def augment_dataset(input_path: str,
                    output_path: str,
                    domain_list: List[str],
                    num_samples: int = 10000):
    """
    add sentence pair for SST task based on current dataset.
    for balance, we should make the number of positive class (label = 1) and the number of negtive class equal.

    @param inpu_path/output_path: should be a text or csv file
    @domain_list: the list of domain you need
    @num_sample: the total number of exmaples you need, default 10k
    """
    input_obj = open(input_path, mode='r', encoding='utf-8')
    output_obj = open(output_path, mode='w', encoding='utf-8')

    data = dict()
    for domain in domain_list:
        data[domain] = []

    for line in input_obj:
        if line == '\n':
            continue
        label = line.split()[0]
        text = line.split()[-1]
        if label in domain_list:
            data[label].append(text)

    for k, v in data.items():
        data[k] = set(v)
        data[k] = list(v)

    num_domain = len(domain_list)
    num_domain_examples = [len(v) for k, v in data.items()]
    num_iter = 0
    class_count = {'pos': 0, 'neg': 0}

    # state_track is a () to record the examples stored by a quadruples,
    # (text_a_label, text_a_idx, text_b_label, text_b_idx)
    state_track = set()

    while num_iter < num_samples:
        text_a_label = randint(0, num_domain - 1)
        text_b_label = randint(0, num_domain - 1)
        text_a_idx = randint(0, num_domain_examples[text_a_label] - 1)
        text_b_idx = randint(0, num_domain_examples[text_b_label] - 1)
        text_a = data[domain_list[text_a_label]][text_a_idx].strip()
        text_b = data[domain_list[text_b_label]][text_b_idx].strip()
        # target = 0 means text_a and text_b are equivalent in semantic
        target = 0
        if text_a_label == text_b_label:
            target = 1
            if text_a_idx == text_b_idx:
                continue

        if class_count[
                'pos'] > class_count['neg'] + 2 and target == 1 or class_count[
                    'neg'] > class_count['pos'] + 2 and target == 0:
            continue

        if (text_a_label, text_a_idx, text_b_label, text_b_idx) in state_track:
            continue
        elif (text_b_label, text_b_idx, text_a_label,
              text_a_idx) in state_track:
            continue

        if target == 1:
            class_count['pos'] += 1
        else:
            class_count['neg'] += 1

        output_obj.writelines(f'{text_a}, {text_b}, {target}\n')
        num_iter += 1
        state_track.add((text_a_label, text_a_idx, text_b_label, text_b_idx))


def convert_to_span_format(input_dir,
                           output_dir,
                           metadata_template_path,
                           tag_list=None):
    """
    convert MSRA likewise NER dataset to span format for matching Minerva fomat.
    version 0.1 use no newlines(/n), i.e., all sentence will be orgnized as a paragraph.
    """
    tags = set(json.loads(tag_list))
    types = ['dev', 'train', 'test']
    files = os.listdir(input_dir)

    ## TODO: optimize code below
    metadata_template = json.load(
        open(metadata_template_path, mode='r', encoding='utf-8'))

    for _file in files:
        _type = _file.split('.')[0]
        if _type not in types:
            continue

        datasets_folder = os.path.join(output_dir, f'{_type}')
        datasets_folder_obj = os.mkdir(datasets_folder)
        f_obj = open(os.path.join(input_dir, _file),
                     mode='r',
                     encoding='utf-8')
        tag_json_obj = None
        span_dict = None
        for i, line in enumerate(f_obj):
            if i % 20 == 0:
                batch_num = i // 20
                dataset_folder = os.path.join(datasets_folder,
                                              _type + f'_{batch_num:0>4}')
                dataset_folder_obj = os.mkdir(dataset_folder)
                metadata_obj = open(os.path.join(dataset_folder,
                                                 'metadata.json'),
                                    mode='w',
                                    encoding='utf-8')
                metadata_template["Named Entity Recognition"][
                    "validation"] = False if _type == "train" else True
                metadata_obj.write(json.dumps(metadata_template))
                raw_txt_obj = open(os.path.join(dataset_folder, 'raw.txt'),
                                   mode='w',
                                   encoding='utf-8')
                # write named_entity_recognition.tag.json of last iteration
                if tag_json_obj:
                    for key in list(span_dict.keys()):
                        if not span_dict[key]:
                            del span_dict[key]
                    tag_json_obj.write(json.dumps(span_dict))
                tag_json_obj = open(os.path.join(
                    dataset_folder, 'named_entity_recognition.tag.json'),
                                    mode='w',
                                    encoding='utf-8')
                # span_dict = dict()
                span_dict = {k: [] for k in tags}
                length = 0
            words = ''  # recode one line
            items = line.strip().split()
            for item in items:
                word = item.split('/')[0]
                tag = item.split('/')[-1]
                word_len = len(word)
                length += word_len
                words += word
                if tag in tags:
                    span = [length - word_len, length]
                    span_dict[tag].append(span)
            raw_txt_obj.write(words + '\n')
            length += 1

            # if i % 20 == 19:


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, help="the input text path")
    parser.add_argument("--output_path", type=str, help="the output text path")
    parser.add_argument("--method",
                        type=str,
                        help="the method you need to use")

    # other optional params
    parser.add_argument("--num_samples",
                        default=20000,
                        type=str,
                        help="choose how many examples you need")
    parser.add_argument("--domain_list",
                        default=['weather', 'flight', 'train', 'map'],
                        type=List[str],
                        help="just look at the name")
    parser.add_argument(
        "--tag_list",
        default='["nr","nt","ns"]',
        # type=List[str],
        help=
        "the tag list you enter should like ['nr', 'nt', 'ns'] without any 'o' tag"
    )
    parser.add_argument(
        "--metadata_template_path",
        default=
        '/home/ubuntu/workspace/data/dataset/yuzhu/电子病例-002/metadata.json',
        # type=List[str],
        help="metadata.json template")

    args = parser.parse_args()

    func_list = {
        'domain_count': domain_count,
        'convert_text_to_json': convert_text_to_json,
        'augment_dataset': augment_dataset,
        'sample': sample,
        'convert_to_span_format': convert_to_span_format,
        'json_to_texts': json_to_texts,
        'de_depulicate': de_depulicate,
        'split': split
    }
    func = args.method
    func = func.lower()

    if func == 'augment_dataset':
        num_samples = int(args.num_samples)
        domain_list = args.domain_list
        # args.domain_list = json.loads(args.domain_list)
        func_list[func](args.input_path, args.output_path, domain_list,
                        num_samples)
    elif func == 'sample':
        func_list[func](args.input_path, args.output_path)
    elif func == 'convert_to_span_format':
        if os.path.isdir(args.input_path) and os.path.isdir(args.output_path):
            if args.tag_list:
                func_list[func](args.input_path, args.output_path,
                                args.metadata_template_path, args.tag_list)
            else:
                raise ValueError('Need tag list not to be empty')
        else:
            raise IOError('Not a correct file path')
    elif func == 'json_to_texts':
        func_list[func](args.input_path)
    elif func == 'de_depulicate':
        func_list[func](args.input_path, args.output_path)
    elif func == 'domain_count':
        func_list[func](args.input_path)
    elif func == 'split':
        func_list[func](args.input_path, args.output_path)