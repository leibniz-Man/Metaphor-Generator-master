import torch
import torch.nn.functional as F

class L3loss:
    def __init__(self):
        super(L3loss, self).__init__()
        self.α = 1
        self.criterion_1 = torch.nn.BCEWithLogitsLoss()
        self.criterion_2 = torch.nn.KLDivLoss(reduction='batchmean')
        self.criterion_3 = torch.nn.CrossEntropyLoss()

    def forward(self, output, attn_score, nmc_score, label, nmi_score=False, nmi_label=False):

        attn_score = F.log_softmax(attn_score,dim=-1)
        nmc_score = F.log_softmax(nmc_score,dim=-1)
        loss2 = self.criterion_2(attn_score, nmc_score)
        loss3 = self.criterion_3(output, label)
        if nmi_score is not False:
            loss1 = self.criterion_1(nmi_score,nmi_label.float())
            return loss1+loss2+loss3
        else:
            return loss2+loss3
