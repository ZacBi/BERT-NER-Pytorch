import logging, os

from flask import Flask, request
from flask_restful import Api, Resource, reqparse

from dataprocess_ner import *
from args import *

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


from pytorch_transformers import (BertTokenizer, BertForSequenceClassification,
                                  BertForTokenClassification, AdamW)
from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()
parser.add_argument('text', type=str, help='The input sentence')

device = torch.device("cuda")
args = None
model = None
tokenizer = None

def load_model():
    """
    Just use one model at one time.
    """
    global model, tokenizer, args

    # tonkenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    # get model
    model = BertForTokenClassification.from_pretrained(
        args.bert_model, num_labels=args.num_labels)
    model.to(device)
    model.eval()


def prepare_data(text):
    """
    For sequence(s)

    @param text: a sentence with better not than 32 tokens (because our max_seq_len is 32)
    
    @return input_ids: input_ids is a tensor of which each element is a id of token in input sentence
    @return input_mask: distinguish the padding and real input
    @return input_type_ids: distingush text_a with text_b, of course we only have text_a
    @return label_list: the classes list
    """

    # processor_dict = {'simpleclasspro': SimpleClassPro, 'stspro': STSPro}
    processor_dict = {'msraprocessor': MSRAProcessor}
    args.task_name = args.task_name.lower()
    processor = processor_dict[args.task_name]()  # detail
    label_list = processor.get_labels()

    text_group = text.strip().split(',')
    if len(text_group) == 1:
        test_example = [InputExample(unique_id=0, text_a=text_group[0].strip())]
    elif len(text_group) == 2:
        test_example = [
            InputExample(unique_id=0,
                         text_a=text_group[0].strip(),
                         text_b=text_group[1].strip())
        ]

    features = convert_examples_to_features(test_example, label_list,
                                            tokenizer, seq_length=32)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_type_ids = torch.tensor(
        [f.input_type_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features],
                                  dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_type_ids,
                              all_input_mask)
    # Run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data, sampler=test_sampler, batch_size=1)


    # no return of label because we don't need
    return test_dataloader, label_list


class Predication(Resource):
    def post(self):
        service_args = parser.parse_args()
        text = service_args['text']
        # print('text')
        pred = self.predict(text)
        return pred, 201

    def predict(self, text):
        if request.method == 'POST':
            ## TODO get the text from request
            print(text)
            test_dataloader, label_list = prepare_data(text)

            # model.eval()
            for input_ids, input_type_ids, input_mask in test_dataloader:
                print(input_ids)
                input_ids = input_ids.to(device)
                input_type_ids = input_type_ids.to(device)
                input_mask = input_mask.to(device)
                text_len = input_mask.sum()

                with torch.no_grad():
                    logits = model(input_ids)[0]
                    pred = logits.max(-1)[-1].squeeze_()
                    seq = ''
                    print(pred)
                    # print(pred.size())
                    # for i in range(int(text_len)):
                    #     print(label_list[int(pred[i])])
                    # pred_label = label_list[pred]

            return 1


api.add_resource(Predication, '/pred')


if __name__ == "__main__":

    args = get_service_args()
    load_model()

    # set debug=False while in production environment
    app.debug = True
    app.run(host='0.0.0.0')
    # app.run()
