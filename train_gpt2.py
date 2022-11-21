import sys
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from process_data.load_cmc_data import *
from process_data.load_pretrain_unlabeled_data import *
from gpt_model import *
from tqdm.auto import tqdm
from transformers import AdamW,get_linear_schedule_with_warmup


def print_log(epoch, epochs):
    print('\n' + '========' * 4 + '[epoch:%d/%d]' % (epoch + 1, epochs) + '========' * 4)


def collate_fn_train(data):
    triple_data = tuple([t['triple'] for t in data])

    triple_ids = pad_sequence(triple_data, batch_first=True, padding_value=0)
    labels = pad_sequence(triple_data, batch_first=True, padding_value=-100)

    return triple_ids[:, :-1], labels[:, 1:]


def train_gpt2():
    args = create_args()
    train_data = DataLoader(metaphor_data(), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_train)
    model = gpt_model().to(torch.device(args.device))
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    total_steps_labeled = (len(train_data) // args.batch_size) * args.epochs
    warm_up_ratio = 0.1

    scheduler_labeled = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_ratio * total_steps_labeled,
                                                        num_training_steps=total_steps_labeled)
    for epoch in range(args.epochs):
        model.train()
        print_log(epoch, args.epochs)
        loss_avg = []
        for ids, (triple, labels) in enumerate(tqdm(train_data, file=sys.stdout)):
            # batch = batch.to(torch.device(args.device))
            labels = labels.to(torch.device(args.device))
            # triple = triple.to(torch.device(args.device))
            triple = triple.to(torch.device(args.device))
            # nmi_label = nmi_label.to(torch.device(args.device))
            # triple = model2(triple)
            output = model(triple)
            loss = criterion(output.view(-1, args.vocab_length), labels.reshape(-1))

            loss.backward()
            optimizer.step()
            scheduler_labeled.step()
            optimizer.zero_grad()
            loss_avg.append(loss.item())
            if (ids + 1) % (len(train_data) // 4) == 0:
                tqdm.write("step:[%d/%d],loss_avg:%.6f" % ((ids + 1), len(train_data), np.mean(loss_avg)))
        torch.save(model, 'model/save_model/model_gpt2_{}.pkl'.format(epoch + 1))


train_gpt2()
