import json
import torch
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from main import *


class self_training_cmc_data(torch.utils.data.Dataset):
    def __init__(self):
        super(self_training_cmc_data, self).__init__()
        args = create_args()
        self.tokenizer = BertTokenizerFast.from_pretrained(args.model_path)
        train_data_path = args.CMC_data_path
        self.train_data = []
        self.train_label = []
        with open(train_data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line)
                self.train_data.append(line['sent'])
                if line['tenor'] != '_' and line['comparator'] != '_' and line['vehicle'] != '_':
                    self.train_label.append(1)
                else:
                    self.train_label.append(0)
        print('pretrain train data preprocess complete')

    def __getitem__(self, item):
        text = self.train_data[item]
        label = self.train_label[item]
        text = self.tokenizer.encode_plus(text, max_length=510, truncation=True, padding='max_length',
                                          return_tensors='pt')

        return {'input_ids': text['input_ids'], 'attention_mask': text['attention_mask'], 'label': label}

    def __len__(self):
        return len(self.train_data)
