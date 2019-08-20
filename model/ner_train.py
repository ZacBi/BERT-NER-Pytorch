"""
Finetuning the library models for NER on MSRA-NER
Author: ZacBi
Version: 0.0.2
Date: 2019/08/06
Attention: 1.remove the usage of XLM;
           2. the reason why not reach SOTA maybe the tokenizer?

"""
# TODO: study the doc
# TODO: save the best performance checkpoint and test on test file finally

import logging
import os
import sys
import random
import glob

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForTokenClassification, BertTokenizer,
                                  XLMConfig, XLMForQuestionAnswering,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForQuestionAnswering, XLNetTokenizer)
from pytorch_transformers import AdamW, WarmupLinearSchedule

sys.path.append('/home/ubuntu/workspace/github/BERT-NER-Pytorch')

# os.environ[
#     'TASK_DATA_PATH'] = '/home/ubuntu/workspace/github/BERT-NER-Pytorch/data/msra_ner'
# os.environ[
#     'MODEL_PATH'] = '/home/ubuntu/workspace/data/model-zoo/ernie_base_128_pytorch'
# os.environ[
#     'OUTPUT_DIR'] = '/home/ubuntu/workspace/github/BERT-NER-Pytorch/outputs'
# os.environ[
#     'WORKSPACE'] = '/home/ubuntu/workspace/github/BERT-NER-Pytorch/model'

from model.dataprocess import *
from model.sequence_eval import *
from model.args import get_train_args

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                  for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForTokenClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer):
    """Train the model"""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.log_dir)
        args.train_batch_size = args.per_gpu_train_batch_size * max(
            1, args.n_gpu)
        train_sampler = RandomSampler(
            train_dataset) if args.local_rank == -1 else DistributedSampler(
                train_dataset)
        train_dataloader = DataLoader(train_dataset,
                                      sampler=train_sampler,
                                      batch_size=args.train_batch_size)
    if args.max_steps > 0:
        t_total = args.max_steps  # t_total: iteration total?
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader
        ) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    # TODO: maybe no_decay is empty in ernie of paddle version,
    # so here we define no_decay as None
    # no_decay = ['bias', 'LayerNorm.weight']
    no_decay = []
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        args.weight_decay
    }, {
        'params': [
            p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        0.0
    }]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    # TODO: learn about shceduler
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=args.warmup_steps,
                                     t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
            pass
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level=args.fp16_opt_level)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps *
        (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    # TODO: what's the difference between t_total and global_step?
    global_step = 0
    # tr_loss: train loss
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs),
                            desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    # Added here for reproductibility (even between python 2 and 3)
    set_seed(args)
    best_f1 = 0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader,
                              desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # Q: why use tuple rather than list?
            batch = tuple(t.to(args.device) for t in batch)
            # FIXME: change the inputs to suit MSRA-NER
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'labels': batch[3]
            }

            ouputs = model(**inputs)
            # model outputs are always tuple in pytorch-transformers (see doc)
            loss = ouputs[0]

            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel (not distributed) training
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer),
                                               args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [
                        -1, 0
                ] and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    # Log metrics
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar(f'{key}/train', value,
                                                 global_step)
                            logger.info(f'{key}/train: {value:.4}')
                    tb_writer.add_scalar('LR/train',
                                         scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('Loss/train',
                                         (tr_loss - logging_loss) /
                                         args.eval_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [
                        -1, 0
                ] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    ckpt_save_dir = os.path.join(args.output_dir,
                                                 f'checkpoint-{global_step}')
                    if not os.path.exists(ckpt_save_dir):
                        os.makedirs(ckpt_save_dir)
                    model_to_save = model.module if hasattr(
                        model, 'module'
                    ) else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(ckpt_save_dir)
                    torch.save(
                        args, os.path.join(ckpt_save_dir, 'training_args.bin'))
                    logger.info(f"Saving model checkpoint to {ckpt_save_dir}")

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args,
                                                          tokenizer,
                                                          evaluate=True,
                                                          output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(
        dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []

    label_count, infer_count, correct_count = 0, 0, 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                # XLM don't use input_type_ids
                'token_type_ids':
                None if args.model_type == 'xlm' else batch[2],
                'labels': batch[3]
            }

            outputs = model(**inputs)
            loss, scores = outputs[:2]

            input_mask = batch[1]
            label_ids = batch[3]
            example_indices = batch[4]
            # np_infers with shape (bath_size=1, max_seq_len)
            np_infers = scores.max(-1)[-1].cpu().numpy()
            np_labels = label_ids.cpu().numpy()
            np_lens = input_mask.sum(-1).cpu().numpy()
            num_label, num_infer, num_correct = chunk_eval(
                np_labels, np_infers, np_lens, args.num_labels)
            # if num_correct > 0 and num_correct == num_label and num_label == num_infer and c:
            #     print(np_labels, '\n', np_infers)
            label_count += num_label
            infer_count += num_infer
            correct_count += num_correct

    precision, recall, f1 = calculate_f1(label_count, infer_count,
                                         correct_count)
    result = {"Precision": precision, "Recall": recall, "F1": f1}
    return result


def load_and_cache_examples(args,
                            tokenizer,
                            evaluate=False,
                            output_examples=False):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    input_file = args.predict_file if evaluate else args.train_file
    cached_features_file = os.path.join(
        os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
            'dev' if evaluate else 'train',
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length)))
    if os.path.exists(cached_features_file
                      ) and not args.overwrite_cache and not output_examples:
    # if os.path.exists(cached_features_file
    #                   ) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info(f"Creating features from dataset file at {input_file}")
        # TODO: [x] modify read_ner_examples()
        examples, label_list = read_ner_examples(task_name=args.task_name,
                                                 input_file=input_file,
                                                 is_training=not evaluate)
        features = convert_examples_to_features(
            examples=examples,
            label_list=label_list,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_length,
            show_example=False)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s",
                        cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features],
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features],
                                  dtype=torch.long)
    all_input_type_ids = torch.tensor([f.input_type_ids for f in features],
                                      dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features],
                                 dtype=torch.long)
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0),
                                         dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask,
                                all_input_type_ids, all_label_ids,
                                all_example_index)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask,
                                all_input_type_ids, all_label_ids)

    if output_examples:
        return dataset, examples, features
    return dataset


def main(args):

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir
    ) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            .format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port),
                            redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1),
        args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path)

    # set the some params of config
    config.num_labels = args.num_labels
    config.layer_norm_eps = args.layer_norm_eps

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name
        if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool('.ckpt' in args.model_name_or_path),
        config=config)

    if args.local_rank == 0:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        # TODO: modify the func below
        train_dataset = load_and_cache_examples(args,
                                                tokenizer,
                                                evaluate=False,
                                                output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(f" global step = {global_step}, average loss = {tr_loss}")

    # Save the trained model and the tokenizer
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        final_save_dir = os.path.join(args.output_dir, 'checkpoint-end')
        if not os.path.exists(final_save_dir):
            os.makedirs(final_save_dir)
        logger.info("Saving model checkpoint to %s", final_save_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(
            model,
            'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(final_save_dir)
        tokenizer.save_pretrained(final_save_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(final_save_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(
                    glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME,
                              recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(
                logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split(
                '-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            result = dict(
                (k + ('_{}'.format(global_step) if global_step else ''), v)
                for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    return results


if __name__ == "__main__":
    main(get_train_args(MODEL_CLASSES))
