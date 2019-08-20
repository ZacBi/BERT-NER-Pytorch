# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Union, List, Dict
import csv
import os
import codecs
import json
import random
import logging
import argparse
import time
from tqdm import tqdm, trange

from .dataprocess import *
from .args import *

from sklearn import metrics
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_transformers import BertTokenizer
from pytorch_transformers import BertForSequenceClassification
from pytorch_transformers.optimization import AdamW
from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def val(model, processor, args, label_list, tokenizer, device):
    '''
    Model evaluation
    
    @param model: 模型
	@param processor: 数据读取方法
	@param args: 参数表
	@param label_list: 所有可能类别
	@param tokenizer: 分词方法
	@param device
	
    @return f1: F1值
    '''
    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, tokenizer, args.max_seq_length, show_exp=False)
    all_input_ids = torch.tensor(
        [f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids, token_type_ids=segment_ids, labels=label_ids)
            loss, logits = outputs[:2]
            pred = logits.max(1)[1]
            predict = np.hstack((predict, pred.cpu().numpy()))
            gt = np.hstack((gt, label_ids.cpu().numpy()))

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()

    print(len(gt))
    f1 = np.mean(metrics.f1_score(predict, gt, average=None))
    # print(f1)

    return f1


def test(model, processor, args, label_list, tokenizer, device):
    '''模型测试
    
    Args:
    model: 模型
	processor: 数据读取方法
	args: 参数表
	label_list: 所有可能类别
	tokenizer: 分词方法
	device

    Returns:
        f1: F1值
    '''
    test_examples = processor.get_test_examples(args.data_dir)
    test_features = convert_examples_to_features(
        test_examples, label_list, tokenizer, args.max_seq_length)
    all_input_ids = torch.tensor(
        [f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data  
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    model.eval()
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
        # input_ids = input_ids.to(device)
        # input_mask = input_mask.to(device)
        # segment_ids = segment_ids.to(device)
        # label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, token_type_ids=segment_ids)[0]
            pred = logits.max(1)[1]
            predict = np.hstack((predict, pred.cpu().numpy()))
            gt = np.hstack((gt, label_ids.cpu().numpy()))

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()

    f1 = np.mean(metrics.f1_score(predict, gt, average=None))
    print('F1 score in text set is {}'.format(f1))

    return f1


def main(args):
    # test the model directly
    if args.do_test:
        args.do_train = False
        args.do_eval = False

    # 对模型输入进行处理的processor，git上可能都是针对英文的processor
    processors = {'simpleclasspro': SimpleClassPro, 'stspro': STSPro}
    # processors = {'stspro': STSPro}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info(
                "16-bits training currently not supported in distributed training")
            # (see https://github.com/pytorch/pytorch/pull/13496)
            args.fp16 = False
    logger.info("device {} n_gpu {} distributed training {}".format(
        device, n_gpu, bool(args.local_rank != -1)))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError(
            "At least one of `do_train` or `do_eval` or 'do_test' must be True.")

    task_name = args.task_name.lower()  # detail
    # print(task_name)

    if task_name not in processors:
        raise ValueError("Task not found: {}".format(task_name))

    processor = processors[task_name]()  # detail
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                          num_labels=2,
                                                          cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))

    if args.fp16:
    	model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[
                                                              args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_())
                           for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_())
                           for n, param in model.named_parameters()]
    else:
    	param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    ## logic: if n is not in no_decay, the weight decay rate of n is set to 0.01,
    ## else: 0.0(no decay truely)
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},  # TODO review weight decay
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    global_step = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, tokenizer, args.max_seq_length, show_exp=False)
        logger.info("***** Running training *****")
        logger.info("  Num examples = {}".format(len(train_examples)))
        logger.info("  Batch size = {}".format(args.train_batch_size))
        logger.info("  Total num steps = {}".format(num_train_steps))
        all_input_ids = torch.tensor(  # TODO Q: where is the position embedding?
            [f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor(
            [f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        ## TODO When we should set shuffle=True in DataLoader()?

        model.train()
        best_score = 0
        flags = 0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            ## TODO the usage of dataloader

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                # outputs = model(input_ids=input_ids, token_type_ids=segment_ids, input_mask, labels=label_ids)
                outputs = model(input_ids=input_ids,
                                token_type_ids=segment_ids, 
                                labels=label_ids)
                loss = outputs[0]
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(
                            param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info(
                                "FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(
                            model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()  # update
                    model.zero_grad()

            ## TODO Look val()
            f1 = val(model, processor, args, label_list, tokenizer, device)
            if f1 > best_score:
                best_score = f1
                print('*f1 score = {}'.format(f1))
                flags = 0
                checkpoint = {
                    'state_dict': model.state_dict()
                }
                # TODO look torch.save()
                model.save_pretrained(args.model_save_pth)
            else:
                print('f1 score = {}'.format(f1))
                flags += 1
                if flags >= 6:  # early stop
                    break
            # torch.save(checkpoint, os.path.join(
            #     args.model_save_pth, 'bert_sst_' + time.strftime("%Y%m%d%H%M",
            #                                                      time.localtime()) + '_epoch{}.pth'.format(epoch)))
            # model.save_pretrained(args.model_save_pth)

    ## TODO review: model save and load
    # model.load_state_dict(torch.load(os.path.join(
    #     args.model_save_pth, 'bert_sst_best.pth'))['state_dict'])
    model = BertForSequenceClassification.from_pretrained(args.model_save_pth,
                                                          num_labels=2,
                                                          cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))
    test(model, processor, args, label_list, tokenizer, device)

if __name__ == '__main__':
	main(get_train_args())
