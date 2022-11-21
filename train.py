import sys

from torch.nn.utils.rnn import pad_sequence
import numpy as np
from process_data.load_cmc_data import *
from process_data.load_pretrain_unlabeled_data import *
from transformers import get_linear_schedule_with_warmup,AdamW
from gpt_model import *
from loss import L3loss
from tqdm.auto import tqdm


def print_log(epoch, epochs):
    print('\n' + '========' * 4 + '[epoch:%d/%d]' % (epoch + 1, epochs) + '========' * 4)


def collate_fn_train(data):
    input_data = tuple([t['data'] for t in data])

    triple_data = tuple([t['triple'] for t in data])
    nmi_label = torch.cat(tuple([t['label'].unsqueeze(0) for t in data]), dim=0)

    input_ids = pad_sequence(input_data, batch_first=True, padding_value=0)

    triple_ids = pad_sequence(triple_data, batch_first=True, padding_value=0)
    labels = pad_sequence(input_data, batch_first=True, padding_value=-100)

    return input_ids[:, :-1], labels[:, 1:], triple_ids, nmi_label


def collate_fn_pretrain(data):
    input_data = tuple([t['data'] for t in data])

    triple_data = tuple([t['triple'] for t in data])

    input_ids = pad_sequence(input_data, batch_first=True, padding_value=0)

    triple_ids = pad_sequence(triple_data, batch_first=True, padding_value=0)
    labels = pad_sequence(input_data, batch_first=True, padding_value=-100)

    return input_ids[:, :-1], labels[:, 1:], triple_ids


def train():
    args = create_args()

    train_data = DataLoader(metaphor_data(), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_train)
    unlabeled_train_data = DataLoader(unlabeled_data(), batch_size=args.batch_size, shuffle=True,
                                      collate_fn=collate_fn_pretrain)
    """预训练部分写完了，下一步是把cmc和未标注数据集放入模型开始训练。后续看一下论文怎么弄loss"""
    model1 = NMG_model().to(torch.device(args.device))
    model1.load_state_dict(torch.load('model/pretrained_model/bart_model.pt',map_location='cuda:0'), strict=False)
    # model2 = lstm_model().to(torch.device(args.device))

    # optimizer = AdamW([{'params': model1.parameters(),'lr':args.lr},{'params': model2.parameters(),'lr':args.lr}])
    optimizer = AdamW([{'params': model1.parameters()}], lr=args.lr)
    criterion = L3loss()

    len_dataset_labeled = len(metaphor_data())
    len_dataset_unlabeled = len(unlabeled_data())
    print('labeled_data:',len_dataset_labeled)
    print('unlabeled_data:',len_dataset_unlabeled)
    total_steps_labeled = (len_dataset_labeled // args.batch_size) * args.epochs
    total_steps_unlabeled = (len_dataset_unlabeled // args.batch_size) * args.epochs
    warm_up_ratio = 0.1

    scheduler_labeled = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_ratio * total_steps_labeled,
                                                        num_training_steps=total_steps_labeled)

    scheduler_unlabeled = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_ratio * total_steps_unlabeled,
                                                          num_training_steps=total_steps_unlabeled)
    # if torch.cuda.device_count() > 1:
    #     model1 = nn.DataParallel(model1, device_ids=[0, 1])
    """formal training stage"""
    for epoch in range(args.epochs):
        model1.train()
        # model2.train()
        print_log(epoch, args.epochs)
        loss_avg = []
        for ids, (batch, labels, triple, nmi_label) in enumerate(tqdm(train_data, file=sys.stdout)):
            batch = batch.to(torch.device(args.device))
            labels = labels.to(torch.device(args.device))
            triple = triple.to(torch.device(args.device))
            nmi_label = nmi_label.to(torch.device(args.device))
            # triple = model2(triple)
            output, attn_score, nmc_score, nmi_score = model1(batch, triple, labels)

            loss = criterion.forward(output.view(-1, output.size(-1)), attn_score, nmc_score, labels.view(-1),
                                     nmi_score, nmi_label)
            loss.backward()
            nn.utils.clip_grad_norm_(model1.parameters(), max_norm=0.01208, norm_type=2)
            optimizer.step()
            scheduler_labeled.step()
            optimizer.zero_grad()

            loss_avg.append(loss.item())
            if (ids + 1) % (len(train_data) // 4) == 0:
                tqdm.write("step:[%d/%d],loss_avg:%.6f" % ((ids + 1), len(train_data), np.mean(loss_avg)))
                loss_avg = []
        if epoch > 2:
            for ids, (batch, labels, triple) in enumerate(tqdm(unlabeled_train_data, file=sys.stdout)):
                batch = batch.to(torch.device(args.device))
                labels = labels.to(torch.device(args.device))
                triple = triple.to(torch.device(args.device))

                # triple = model2(triple)
                output, attn_score, nmc_score = model1(batch, triple, labels, self_training=True)

                loss = criterion.forward(output.view(-1, output.size(-1)), attn_score, nmc_score, labels.view(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(model1.parameters(), max_norm=0.01208, norm_type=2)
                optimizer.step()
                scheduler_unlabeled.step()
                optimizer.zero_grad()

                loss_avg.append(loss.item())
                if (ids + 1) % (len(unlabeled_train_data) // 4) == 0:
                    tqdm.write("step:[%d/%d],loss_avg:%.6f" % ((ids + 1), len(unlabeled_train_data), np.mean(loss_avg)))
                    loss_avg = []
        torch.save(model1, 'model/save_model/model_bart_{}_2.pkl'.format(epoch + 1))
        # torch.save(model2, 'model/save_model/model_lstm_{}_1.pkl'.format(epoch + 1))


train()
