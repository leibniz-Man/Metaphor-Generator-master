import torch
from transformers import logging, BertTokenizer
from main import *

logging.set_verbosity_error()

args = create_args()
model1 = torch.load(args.saved_model_path + '/model_bart_5_1.pkl').to(torch.device(args.device))
model2 = torch.load(args.saved_model_path + '/model_gpt2_11.pkl').to(torch.device(args.device))
# model3 = torch.load(args.saved_model_path + '/model_lstm_9_1.pkl').to(torch.device(args.device))

model1.eval()
model2.eval()
# model3.eval()

tokenizer = BertTokenizer.from_pretrained(args.bart_model_path, sep_token='[SEP]', cls_token='[CLS]', pad_token='[PAD]',
                                          mask_token='[MASK]')

while True:
    i = input('请输入:')

    data = tokenizer.encode(i, max_length=300, truncation=True, padding=False, add_special_tokens=False,
                            return_tensors='pt')
    data = data[-1].to(torch.device(args.device))
    triple = tokenizer.encode(tokenizer.cls_token, max_length=300, truncation=True, padding=False,
                              add_special_tokens=False,
                              return_tensors='pt')

    un_triple = model2(data, is_train=False)
    for pre_triple in un_triple:
        un_triple = tokenizer.decode(pre_triple.squeeze(0)[:-1], skip_special_tokens=True)
        print('predicted_triple:', un_triple)

        un_triple = tokenizer.encode(un_triple, max_length=300, truncation=True, padding=False,
                                     add_special_tokens=False,
                                     return_tensors='pt')
        un_triple = un_triple.to(torch.device(args.device))
        # un_triple = model3(un_triple)
        output = model1(
            x=triple.long().to(torch.device(args.device)),
            triple=un_triple.to(torch.device(args.device))
        )
        for sentence in output:
            sentence = tokenizer.decode(sentence.squeeze(0)[:-1], skip_special_tokens=True)
            # ids = sentence.index('。')
            # print(sentence[:ids+1])
            print('sentence:', sentence)
