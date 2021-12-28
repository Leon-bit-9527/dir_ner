import numpy as np
import torch
from torch import nn
from architectures.LSTM_NER import LSTM_NER
from architectures.transformer import TransformerEncoder
from architectures.wordrep import WordRep 
import torch.optim as optim
from torch.distributions.dirichlet import Dirichlet
from torch.nn.utils.clip_grad import clip_grad_norm_
from .datahelper import get_one_hot, batchify_with_label, recover_label, random_embedding_label, generate_label_mask
# from .NormalizingFlowDensity import NormalizingFlowDensity
import torch.nn.functional as F
from .metric import confidence
from utils.optimizer import *

__density_types__ = {'planar_flow': None,
                     'radial_flow': None,
                     'iaf_flow': None,
                     'normal_mixture': None}

__budget_functions__ = {'one': lambda N: torch.ones_like(N),
                        'log': lambda N: torch.log(N + 1.),
                        'id': lambda N: N,
                        'id_normalized': lambda N: N / N.sum(),
                        'exp': lambda N: torch.exp(N),
                        'parametrized': lambda N: torch.nn.Parameter(torch.ones_like(N).to(N.device))}


class PosteriorN(nn.Module):
    def __init__(self, data):  # Budget function name applied on class count. name
        super().__init__()
        torch.set_default_tensor_type(torch.FloatTensor)
        self.gpu = data.HP_gpu

        # architectures parameters
        self.threshold = data.HP_threshold
        self.latent_dim = data.latent_dim
        self.k_lipschitz = data.k_lipschitz
        self.density = data.density
        self.budget_function = data.budget_function
        self.N = data.label_alphabet_size
        self.label_alphabet = data.label_alphabet
        self.output_dim = data.label_alphabet_size
        self.wordrep = WordRep(data)
        self.architectures = LSTM_NER(data)
        self.label_embedding = nn.Embedding(data.label_alphabet_size, data.HP_label_embed_dim)
        self.label_embedding.weight.data.copy_(torch.from_numpy(
            random_embedding_label(data.label_alphabet_size, data.HP_label_embed_dim, data.HP_label_embedding_scale)))
        
        self.word2hidden = nn.Linear(self.wordrep.total_size, data.d_architectures)
        self.label2hidden = nn.Linear(data.HP_label_embed_dim, data.d_architectures)

        self.encoder = TransformerEncoder(data.HP_architectures2_layer, data.d_architectures, data.HP_nhead,
                                          data.HP_dim_feedforward,dropout=data.HP_architectures2_dropout, dropout_attn=data.HP_attention_dropout,
                                          )

        self.model2_fc_dropout = nn.Dropout(data.HP_architectures2_dropout)
        self.t_hidden2tag = nn.Linear(data.d_architectures * 2, data.label_alphabet_size)

        if self.budget_function in __budget_functions__:
            self.N, self.budget_function = __budget_functions__[self.budget_function](self.N), self.budget_function
        else:
            raise NotImplementedError

        # Training parameters
        self.batch_size, self.lr = data.HP_batch_size, data.HP_lr
        self.loss, self.regr = data.HP_loss, data.HP_regr

        self.batch_norm = nn.BatchNorm1d(num_features=self.latent_dim)
        self.softmax = nn.Softmax(dim=-1)

        self.m2_params = [self.t_hidden2tag, self.wordrep, self.word2hidden, self.label2hidden, self.encoder,  self.label_embedding, self.softmax]

        if self.gpu:
            for i in range(len(self.m2_params)):
                self.m2_params[i] = self.m2_params[i].cuda()

        ## optimizer
        lr_detail1 = [{"params": filter(lambda p: p.requires_grad, self.architectures.parameters()), "lr": data.HP_lr},
                    ]
        if data.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(lr_detail1,
                                momentum=data.HP_momentum, weight_decay=data.HP_l2,lr=data.HP_lr)
        elif data.optimizer.lower() == "adagrad":
            self.optimizer = optim.Adagrad(lr_detail1, weight_decay=data.HP_l2,lr=data.HP_lr)
        elif data.optimizer.lower() == "adadelta":
            self.optimizer = optim.Adadelta(lr_detail1, weight_decay=data.HP_l2,lr=data.HP_lr)
        elif data.optimizer.lower() == "rmsprop":
            self.optimizer = optim.RMSprop(lr_detail1, weight_decay=data.HP_l2,lr=data.HP_lr)
        elif data.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(lr_detail1, weight_decay=data.HP_l2,lr=data.HP_lr)
        else:
            print("Optimizer illegal: %s" % (data.optimizer))
            exit(1)
        clip_grad_norm_(self.parameters(), data.clip_grad)


        ## model 2 optimizer
        self.optimizer2 = AdamW(self.get_m2_params(), lr=data.HP_lr2, weight_decay=data.HP_l2)
        total_batch = len(data.train_Ids) // data.HP_batch_size + 1
        t_total = total_batch * data.HP_iteration   
        warmup_step = int(data.warmup_step * t_total)
        self.scheduler2 = WarmupLinearSchedule(self.optimizer2, warmup_step, t_total)



    def forward(self, instance, soft_output=None, return_output='alpha', compute_loss=False):

        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(
                instance, self.gpu, True)
        batch_size = len(batch_word)
        mask = mask.eq(1)

        if self.budget_function == 'parametrized':
            N = self.N / self.N.sum()  # To check
        else:
            N = self.N

        zk,logits = self.architectures(batch_word,
            batch_features,
            batch_wordlen,
            batch_char,
            batch_charlen,
            batch_charrecover,)
        alpha1 = torch.exp(logits)
        soft_output_pred = self.softmax(logits)  # shape: [batch_size, output_dim]
        output_pred = self.predict(soft_output_pred, mask)
        correct, p, epis_score, alea_score = confidence(batch_label, alpha1)
        label_mask = generate_label_mask(alea_score, mask, threshold=self.threshold)
        ## model2
        model2_input_label_embed = torch.einsum("bsc,cd->bsd", [p.detach(), self.label_embedding.weight]).masked_fill(mask.unsqueeze(-1)==0, 0)

        word_represent = self.wordrep(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                      batch_charrecover)

        word_represent = self.word2hidden(word_represent)
        model2_input_label_embed = self.label2hidden(model2_input_label_embed)

        hh, hl = self.encoder(word_represent, model2_input_label_embed, mask)

        outs2 = self.t_hidden2tag(self.model2_fc_dropout(torch.cat([hh, hl], -1)))

        model2_preds = self.predict(outs2, mask)

        predicted_seq = output_pred.masked_fill(label_mask==1, 0) + model2_preds.masked_fill(label_mask==0, 0)

        pred_label, gold_label = recover_label(predicted_seq, batch_label, mask, self.label_alphabet, batch_wordrecover)

        # Loss
        if compute_loss:
            if self.loss == 'CE':
                self.grad_loss = self.CE_loss(soft_output_pred, batch_label)
            elif self.loss == 'UCE':
                self.grad_loss = self.UCE_loss(alpha1, outs2, batch_label)
            else:
                raise NotImplementedError

        if return_output == 'train':
            return output_pred
        elif return_output == 'soft':
            return soft_output_pred
        elif return_output == 'dev':
            return correct.tolist(), epis_score.cpu().detach().numpy(), alea_score.cpu().detach().numpy(), pred_label, gold_label
        elif return_output == 'latent':
            return zk
        else:
            raise AssertionError

    def CE_loss(self, soft_output_pred, soft_output):
        # with autograd.detect_anomaly():
        CE_loss = - torch.sum(soft_output.squeeze() * torch.log(soft_output_pred))

        return CE_loss

    def UCE_loss(self, alpha1, outs2, batch_label):
        soft_output = get_one_hot(batch_label, self.gpu, self.N)

        loss1 = self.get_loss1_UCE(alpha1, soft_output)
        loss2 = self.get_loss_CE(outs2, batch_label)

        loss = loss1 + loss2

        return loss
    
    def get_loss_CE(self, outs, batch_label):
        batch_size, seq_len = outs.size()[:2]

        loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
        loss = loss_function(outs.view(batch_size * seq_len, -1), batch_label.view(batch_size * seq_len))

        return loss / batch_size

    def get_loss1_UCE(self, alpha, soft_output):
        out_dim = alpha.size(1)
        alpha_0 = alpha.sum(1).unsqueeze(1).repeat(1, out_dim, 1).view(-1,out_dim,self.N)
        # alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, self.output_dim)
        entropy_reg = Dirichlet(alpha).entropy()
        UCE_loss = torch.sum(soft_output * (torch.digamma(alpha_0) - torch.digamma(alpha))) - self.regr * torch.sum(entropy_reg)

        loss = UCE_loss/self.batch_size

        return loss

    def predict(self, p, mask):
        output_pred = torch.max(p, dim=-1)[1]
        output_pred = output_pred.masked_fill(mask==0, 0)  # mask padding words

        return output_pred

    def get_m2_params(self):
        return nn.ModuleList(self.m2_params).parameters()

    def step(self):
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        self.grad_loss.backward()
        self.scheduler2.step()
        self.optimizer.step()
        self.optimizer2.step()

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print("Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def add_dropout(x, dropout):
    ''' x: batch * seq_len * hidden '''
    return F.dropout2d(x.transpose(1,2)[...,None], p=dropout, training=True).squeeze(-1).transpose(1,2)
