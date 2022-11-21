import sys
from process_data.load_cmc_data import *
from process_data.load_pretrain_cmc_data import *
from process_data.load_pretrain_unlabeled_data import *
from gpt_model import *


def print_log(epoch, epochs):
    print('\n' + '========' * 4 + '[epoch:%d/%d]' % (epoch + 1, epochs) + '========' * 4)

def train():
    args = create_args()

    pretrain_train_data = DataLoader(self_training_cmc_data(), batch_size=args.batch_size, shuffle=True)
    unlabeled_test_data = DataLoader(unlabeled_data(), batch_size=args.batch_size, shuffle=False)
    model = bart_model().to(torch.device(args.device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(3):
        model.train()
        print_log(epoch, args.epochs)
        acc_avg = 0
        for ids, batch in enumerate(tqdm(pretrain_train_data, file=sys.stdout)):
            input_ids = batch['input_ids'].squeeze(1).to(torch.device(args.device))
            attention_mask = batch['attention_mask'].squeeze(1).to(torch.device(args.device))
            label = batch['label'].to(torch.device(args.device))
            output = model(input_ids,attention_mask).squeeze(1)
            loss = criterion(output,label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            k = sum(torch.eq(torch.sigmoid(output).round(),label).tolist())
            acc_avg += k
        print(acc_avg/(len(pretrain_train_data)*args.batch_size))

    torch.save(model.state_dict(),'bart_model.pt')

train()