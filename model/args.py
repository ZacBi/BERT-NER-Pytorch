import argparse

def get_train_args():
    parser = argparse.ArgumentParser()

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument("--model_save_dir",
                        default='/home/ubuntu/workspace/data/checkpoint/chatbot',
                        type=str,
                        #required = True,
                        help="The output **directory** where the model checkpoints will be written")
    parser.add_argument("--do_test",
                        default=False,
                        help='test some text directly')
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="训练模式")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="验证模式")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="英文字符的大小写转换，对于中文来说没啥用")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="训练时batch大小")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="Adam初始学习步长")
    parser.add_argument("--num_train_epochs",
                        default=12.0,
                        type=float,
                        help="训练的epochs次数")
    parser.add_argument("--warmup_proportion",  # TODO learn about warm up tech
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for."
                        "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',  # TODO argparser: action
                        help="don't use CUDA")
    parser.add_argument("--seed",
                        default=777,
                        type=int,
                        help="初始化时的随机数种子")
    parser.add_argument("--gradient_accumulation_steps",  # TODO why pytorch clear gradient for every batch: gradient accumulation?
                        default=1,
                        type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    args = parser.parse_args()
    return args

def get_test_args():
    parser = argparse.ArgumentParser()

    add_common_args(parser)
    add_train_test_args(parser)
    parser.add_argument(
        "--text",
        default=
        "日本知名学者石川一成先生曾撰文说：面对宝顶大佛湾造像，看中华民族囊括外来文化的能力和创造能力，不禁使我目瞪口呆。",
        type=str)
    # parser.add_argument("--state_dict",
    #                     default="/home/ubuntu/workspace/data/checkpoint/chatbot/bert_sst_best.pth",
    #                     type=str,
    #                     help="the state_dict, always be the best state dict has been trained")

    args = parser.parse_args()
    return args

def get_service_args():
    parser = argparse.ArgumentParser()

    add_common_args(parser)
    # parser.add_argument("--state_dict",
    #                     default="/home/ubuntu/workspace/data/checkpoint/chatbot/bert_sst_best.pth",
    #                     type=str,
    #                     help="the state_dict, always be the best state dict has been trained")


    args = parser.parse_args()
    return args

def add_common_args(parser):
    """
    Common args for train, test and service.
    """

    parser.add_argument("--bert_model",
                        default='/home/ubuntu/.cache/torch/pytorch_transformers',
                        type=str,
                        #required = True,
                        help="choose [bert-base-chinese] mode, of which pertrain model has been sotred.")
    parser.add_argument("--task_name",
                        default='SimpleClassPro',
                        type=str,
                        #required = True,
                        help="The processor you need for handle examples")
    parser.add_argument("--num_labels", default=9, type=int)


def add_train_test_args(parser):

    parser.add_argument("--data_dir",
                        default='/home/ubuntu/workspace/data/dataset/fake',
                        type=str,
                        #required = True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The max sequence length for train, test and others")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="验证时batch大小")

