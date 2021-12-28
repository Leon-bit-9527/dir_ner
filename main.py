# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import sys
import argparse
import random
import torch
import gc
import torch.optim as optim
import numpy as np
from model.posterior_networks.metric import get_ner_fmeasure
from utils.data import Data
from torch.nn.utils.clip_grad import clip_grad_norm_
from utils.optimizer import *
from utils.metrics_prior import accuracy, confidence, brier_score, anomaly_detection
from model.posterior_networks.PosteriorN import PosteriorN
import pickle
from model.posterior_networks.metric import auc
import os


def data_initialization(data):
    # data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir)
    data.build_alphabet(data.dev_dir)
    data.build_alphabet(data.test_dir)
    data.fix_alphabet()

def evaluate(data, model, name):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'OOD_test':
        instances = data.OOD_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
        exit(1)

    corrects = []
    epis_scores = []
    alea_scores = []
    pred_results = []
    gold_results = []
    batch_size = data.HP_batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1
    gold_label = instances[3]
    model.eval()

    with torch.no_grad():

        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = instances[start:end]
            if not instance:
                continue

            correct, epis_score, alea_score, pred_label, gold_label = model(instance, return_output='dev', compute_loss=False)
            
            corrects.append(correct)
            epis_scores.append(epis_score)
            alea_scores.append(alea_score)
            pred_results += pred_label
            gold_results += gold_label

    decode_time = time.time() - start_time
    speed = len(instances) / decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    # epis_auc = auc(corrects, epis_scores)
    # alea_auc = auc(corrects, alea_scores)

    if data.seg:
        score = f
        print("%s: time: %.2f s, speed: %.2f doc/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f; \n" %
              (name, decode_time, speed, acc, p, r, f))
        # print("epis_auc: %.4f, alea_auc: %.4f\n" %
        #       (epis_auc, alea_auc))
    else:
        score = acc
        print("%s: time: %.2f s speed: %.2f doc/s; acc: %.4f; \n" % (name, decode_time, speed, acc))

    if name == 'raw':
        print("save predicted results to %s" % data.decode_dir)
        # data.write_decoded_results(pred_results,total_uncertainty,prob,name)

    return score, pred_results

def train(data):
    print("Training model...")
    data.show_data_summary()
    save_data_name = data.model_dir + "/data.dset"
    if data.save_model:
        data.save(save_data_name)

    batch_size = data.HP_batch_size
    train_num = len(data.train_Ids)
    total_batch = train_num // batch_size + 1

    model = PosteriorN(data)
    pytorch_total_params = sum(p.numel() for p in model.parameters())

    print(model)
    print("pytorch total params: %d" % pytorch_total_params)

    best_dev = -10
    best_test = -10
    max_test = -10
    max_test_epoch = -1
    max_dev_epoch = -1

    ## start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        print("\n ###### Epoch: %s/%s ######" % (idx, data.HP_iteration))  # print (self.train_Ids)

        sample_loss = 0
        total_loss = 0
        random.shuffle(data.train_Ids)

        model.train()

        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            output_pred = model(instance, return_output='train', compute_loss=True)
            loss = model.grad_loss.item()
            sample_loss += loss
            total_loss += loss
            if end % 500 == 0:
                if sample_loss > 1e8 or str(sample_loss) == "nan":
                    loss = 0
                    print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                sys.stdout.flush()
                sample_loss = 0

            model.step()

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2f s, speed: %.2f doc/s,  total loss: %s" % (
            idx, epoch_cost, train_num / epoch_cost, total_loss))

        # dev
        dev_score, _ = evaluate(data, model, "dev")

        # test
        test_score, _ = evaluate(data, model, "test")

        if max_test < test_score:
            max_test_epoch = idx
        max_test = max(test_score, max_test)
        if dev_score > best_dev:
            print("Exceed previous best dev score")
            best_test = test_score
            best_dev = dev_score
            max_dev_epoch = idx
            if data.save_model:
                model_name = data.model_dir + "/best_model.ckpt"
                print("Save current best model in file:", model_name)
                torch.save(model.state_dict(), model_name)

        print("Score summary: max dev (%d): %.4f, test: %.4f; max test (%d): %.4f" % (
            max_dev_epoch, best_dev, best_test, max_test_epoch, max_test))

        gc.collect()


def load_model_decode(data):
    print("Load Model from dir: ", data.model_dir)
    model = PosteriorN(data)
    model_name = data.model_dir + "/best_model.ckpt"
    model.load_state_dict(torch.load(model_name, map_location='cpu'))

    evaluate(data, model, "raw")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with UANet')
    parser.add_argument('--config', help='Configuration File')

    parser.add_argument('--train_dir', default='data/conll2003/train.txt', help='train_file')
    parser.add_argument('--dev_dir', default='data/conll2003/dev.txt', help='dev_file')
    parser.add_argument('--test_dir', default='data/conll2003/test.txt', help='test_file')
    parser.add_argument('--raw_dir', default='data/conll2003/test.txt')
    parser.add_argument('--model_dir', default='outs/onto', help='model_file')

    parser.add_argument('--seg', default=True)
    parser.add_argument('--save_model', default=True)

    parser.add_argument('--word_emb_dir', default=None, help='word_emb_dir')
    parser.add_argument('--norm_word_emb', default=False)
    parser.add_argument('--norm_char_emb', default=False)
    parser.add_argument('--number_normalized', default=True)
    parser.add_argument('--latent_dim', default=10)
    parser.add_argument('--k_lipschitz', default=False)
    parser.add_argument('--density', default=False)
    parser.add_argument('--budget_function', default='id')
    parser.add_argument('--HP_loss', default='UCE')
    parser.add_argument('--HP_regr', default=1e-5)

    # training setting
    parser.add_argument('--status', choices=['train', 'decode'], default='train')
    parser.add_argument('--iteration', default=100)
    parser.add_argument('--batch_size', default=10)
    parser.add_argument('--ave_batch_loss', default=False)
    parser.add_argument('--seed', default=920)

    # word representation
    parser.add_argument('--use_char', default=True)
    parser.add_argument('--char_emb_dim', default=30)
    parser.add_argument('--char_seq_feature', choices=['CNN', 'GRU', 'LSTM'], default='CNN')
    parser.add_argument('--char_hidden_dim', default=50, help="number of filter")
    parser.add_argument('--word_emb_dim', default=100)
    parser.add_argument('--dropout', default=0.5)

    # model1 parameter
    parser.add_argument('--bayesian_lstm_dropout', default=0.01)
    parser.add_argument('--architectures1_dropout', default=0.25)
    parser.add_argument('--hidden_dim', default=400)
    parser.add_argument('--architectures1_layer', default=1)
    parser.add_argument('--bilstm', default=True)

    parser.add_argument('--threshold', default=0.55)

    # model2 parameter
    parser.add_argument('--label_embed_dim', default=400)
    parser.add_argument('--label_embedding_scale', default=600)
    parser.add_argument('--architectures2_layer', default=2)
    parser.add_argument('--d_head', default=80)
    parser.add_argument('--n_head', default=7)
    parser.add_argument('--architectures2_dropout', default=0.2)
    parser.add_argument('--attention_dropout', default=0.15)

    parser.add_argument('--use_crf', default=False)

    # optimizer
    parser.add_argument('--clip_grad', default=5)
    parser.add_argument('--l2', default=1e-7)
    ## model1 optimizer
    parser.add_argument('--optimizer', default='SGD')
    parser.add_argument('--learning_rate', default=0.005)
    parser.add_argument('--lr_decay', default=0.05)
    parser.add_argument('--momentum', default=0.9)
    ## model2 optimizer
    parser.add_argument('--warmup_step', default=0.1)
    parser.add_argument('--learning_rate2', default=0.0001)

    args = parser.parse_args()

    seed_num = int(args.seed)
    print("Seed num:", seed_num)
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    torch.random.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)

    data = Data()
    data.HP_gpu = torch.cuda.is_available()

    if args.status == 'train':
        print("MODE: train")
        data.read_config(args)

        import uuid

        uid = uuid.uuid4().hex[:6]
        data.model_dir = data.model_dir + "_" + uid
        print("model dir: %s" % uid)
        if not os.path.exists(data.model_dir):
            os.mkdir(data.model_dir)

        data_initialization(data)
        data.generate_instance('train')
        data.generate_instance('dev')
        data.generate_instance('test')
        data.build_pretrain_emb()
        train(data)
        print("model dir: %s" % uid)
    elif args.status == 'decode':
        print("MODE: decode")
        data.load(args.model_dir + "/data.dset")
        data.read_config(args)
        data.show_data_summary()
        data.generate_instance('raw')

        load_model_decode(data)
