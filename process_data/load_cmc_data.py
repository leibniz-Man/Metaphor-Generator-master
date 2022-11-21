import json
import torch
from transformers import BertTokenizerFast, logging
from torch.utils.data import DataLoader
from main import *

logging.set_verbosity_error()


class metaphor_data(torch.utils.data.Dataset):
    def __init__(self):
        super(metaphor_data, self).__init__()
        args = create_args()
        self.tokenizer = BertTokenizerFast.from_pretrained(args.bart_model_path)
        train_data_path = args.CMC_data_path
        self.train_data = []
        self.label = []
        self.triple = []
        with open(train_data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line)
                self.train_data.append(line['sent'])
                self.triple.append([line['tenor'], line['comparator'], line['vehicle']])
                if line['tenor'] != '_' and line['comparator'] != '_' and line['vehicle'] != '_':
                    self.label.append(1)
                else:
                    self.label.append(0)
        print('train data preprocess complete')

    def __getitem__(self, item):
        data = self.train_data[item]
        label = self.label[item]
        triple = self.triple[item]
        triple = triple[0] + triple[1] + triple[2]

        data = self.tokenizer.encode(data, max_length=200, truncation=True, padding=False, add_special_tokens=True,
                                     return_tensors='pt')
        triple = self.tokenizer.encode(triple, max_length=200, truncation=True, padding=False,
                                       add_special_tokens=False, return_tensors='pt')
        data = data.squeeze(0).tolist()
        # data.append(self.tokenizer.sep_token_id)
        data = torch.tensor(data)
        label = torch.tensor(label)
        return {'data': data, 'triple': triple.squeeze(0), 'label': label}

    def __len__(self):
        return len(self.train_data)
