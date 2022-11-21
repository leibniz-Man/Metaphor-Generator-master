import torch
import torch.nn as nn
from transformers import logging, BartForConditionalGeneration, BartConfig, GPT2Config, GPT2LMHeadModel
from transformers.file_utils import ModelOutput
from main import *


class bart_model(nn.Module):
    def __init__(self):
        super(bart_model, self).__init__()
        self.args = create_args()
        self.config = BartConfig.from_pretrained(self.args.bart_config_path)
        self.model2 = BartForConditionalGeneration.from_pretrained(self.args.bart_model_path)
        self.dropout = torch.nn.Dropout(self.config.dropout)
        self.mlp = torch.nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        output = self.model2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask)
        output = output['decoder_hidden_states'][-1][:, -1, :]
        output = self.mlp(output)
        return output


class NMG_model(nn.Module):
    def __init__(self):
        super(NMG_model, self).__init__()
        self.args = create_args()
        self.config = BartConfig.from_pretrained(self.args.bart_config_path)
        # self.model1 = BartForConditionalGeneration.from_pretrained(self.args.bart_model_path)
        self.model2 = BartForConditionalGeneration.from_pretrained(self.args.bart_model_path)
        self.dropout = torch.nn.Dropout(self.config.dropout)
        self.mlp = torch.nn.Linear(self.config.hidden_size, 1)
        self.nmc_mlp = torch.nn.Linear(self.config.hidden_size, 1)
        self.query = torch.nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.key = torch.nn.Linear(self.config.hidden_size, self.config.hidden_size)

    def forward(self, x, triple, label=None, self_training=False):
        """formal_training_stage"""
        # encoder_output = self.model1(x)
        # encoder_output = self.model1(input_ids=un_triple)
        # encode_input = encoder_output

        # encoder_output = ModelOutput()
        # encoder_output['last_hidden_state'] = un_triple
        # encoder_output.last_hidden_state = un_triple

        if label is not None:
            encode_output = self.model2(input_ids=triple, decoder_input_ids=x, labels=label)
        else:
            encode_output = self.model2.generate(input_ids=triple, decoder_input_ids=x, num_beams=12,
                                                 max_length=300, num_return_sequences=1)
            return encode_output
        logits = encode_output['logits']
        decoder_output = encode_output['decoder_hidden_states'][-1]
        last_decoder_output = decoder_output[:, -1, :]
        nmi_score = self.mlp(last_decoder_output).unsqueeze(-1)

        """NM Components Identification Output"""
        query = self.query(decoder_output)
        key = self.key(last_decoder_output).unsqueeze(-1)
        """attention score"""

        attn_score = torch.softmax(torch.bmm(query, key) / pow(self.config.hidden_size, 0.5), dim=-2)
        """nm component score"""
        nmc_output = torch.sigmoid(self.nmc_mlp(decoder_output))

        decoder_output_weight = torch.mul(attn_score, nmc_output)

        logits = torch.mul(decoder_output_weight, logits)
        logits = torch.mul(nmi_score, logits)
        output = logits
        if self_training:
            return output, attn_score, nmc_output
        else:
            return output, attn_score, nmc_output, nmi_score.squeeze(-1).squeeze(-1)


class gpt_model(nn.Module):
    def __init__(self):
        super(gpt_model, self).__init__()
        self.args = create_args()
        self.config = GPT2Config.from_pretrained(self.args.gpt_config_path)
        self.model1 = GPT2LMHeadModel.from_pretrained(self.args.gpt_model_path)
        self.dropout = torch.nn.Dropout(self.config.dropout)

    def forward(self, x, is_train=True):
        output = self.model1(x)
        if is_train:
            """batch_size * sequence_length * hidden_size"""
            last_hidden_state = output['hidden_states'][-1]
            logits = output['logits']
            return logits
        else:
            output = self.model1.generate(input_ids=x.unsqueeze(0), num_beams=12, max_length=15, num_return_sequences=10)
            return output


class lstm_model(nn.Module):
    def __init__(self):
        super(lstm_model, self).__init__()
        self.args = create_args()
        self.embedding = torch.nn.Embedding(self.args.vocab_length, 300)
        self.lstm = torch.nn.LSTM(input_size=300, batch_first=True,hidden_size=768)

    def forward(self, x):
        embedding = self.embedding(x)
        output, (_, _) = self.lstm(embedding)
        return output
