import argparse


def get_train_args(MODEL_CLASSES):
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file",
                        default=None,
                        type=str,
                        required=True,
                        help="MSRA tsv for training. E.g., train.tsv")
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        required=True,
        help="MSRA-NER tsv for predictions. E.g., dev.tsv or test.tsv")
    parser.add_argument("--model_type",
                        default=None,
                        type=str,
                        required=True,
                        help="Model type selected in the list: " +
                        ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
    )  #+ ", ".join(ALL_MODELS))
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The output directory where the model checkpoints and predictions will be written."
    )

    ## Other parameters
    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="The directory to save event during train or test")
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help=
        "Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--task_name", default=None, type=str)
    parser.add_argument("--num_labels", default=2, type=int)
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help=
        "The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded."
    )
    parser.add_argument(
        "--do_train",
        action='store_true',
        # default=False,
        help="Whether to run training.")
    parser.add_argument(
        "--do_eval",
        action='store_true',
        # default=False,
        help="Whether to run eval on the dev set without or after train stage."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action='store_true',
        help="Rul evaluation during training at each logging step.")
    parser.add_argument(
        "--do_lower_case",
        action='store_true',
        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size",
                        default=16,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size",
                        default=1,
                        type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        "--layer_norm_eps",
        default=1e-5,
        type=float,
        help="The epsilon used by LayerNorm. In ernie, it's 1e-5")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default=6.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs."
        # good details above
    )
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--verbose_logging",
        action='store_true',
        help=
        "If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal MSRA-NER evaluation.")

    parser.add_argument('--eval_steps',
                        type=int,
                        default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps',
                        type=int,
                        default=100000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action='store_true',
        default=False,
        help=
        "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number"
    )
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir',
                        action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument(
        '--overwrite_cache',
        action='store_true',
        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed',
                        type=int,
                        default=2019,
                        help="random seed for initialization")

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        '--fp16',
        action='store_true',
        help=
        "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"
    )
    parser.add_argument(
        '--fp16_opt_level',
        type=str,
        default='O1',
        help=
        "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip',
                        type=str,
                        default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--server_port',
                        type=str,
                        default='',
                        help="Can be used for distant debugging.")
    args = parser.parse_args()

    return args