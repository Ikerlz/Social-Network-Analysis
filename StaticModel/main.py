#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PP_GNN
@File    ：main.py
@Author  ：Iker Zhe
@Date    ：2022/1/8 16:02
"""
import torch
import torch.nn.functional as F
from model import GraphSAGEModel, GNNRec, FISM
from utils import RecEvaluate_GnnRec, RecEvaluate_FISM
from data_load import *
import argparse
from tqdm import tqdm


def main_loop(model_name='fism'):
    if model_name == "fism":
        gconv_p = GraphSAGEModel(n_hidden,
                                 n_hidden,
                                 n_hidden,
                                 n_layers,
                                 F.relu,
                                 dropout,
                                 aggregator_type)

        gconv_q = GraphSAGEModel(n_hidden,
                                 n_hidden,
                                 n_hidden,
                                 n_layers,
                                 F.relu,
                                 dropout,
                                 aggregator_type)

        model = FISM(
            user_movie_spm=user_movie_spm,
            gconv_p=gconv_p,
            gconv_q=gconv_q,
            in_feats=in_feats,
            num_hidden=n_hidden,
            beta=0,
            gamma=0
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Training
        loss_list = []
        pbar = tqdm(range(n_epochs))
        prev_acc = 0
        test_acc_list = []
        for epoch in pbar:
            model.train()
            loss = model(g, features, neg_sample_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()))

            acc = RecEvaluate_FISM(model, g, features, users_valid, movies_valid, data)
            pbar.set_description('epoch: %d, loss: %.4f, acc_valid: %.4f' % (epoch, loss.item(), acc))
            loss_list.append(loss.item())
            if acc > prev_acc:
                # Save Model
                torch.save(model, model_save_path)
                prev_acc = acc
            test_acc = RecEvaluate_FISM(model, g, features, users_test, movies_test, data)
            test_acc_list.append(test_acc)


    elif model_name == "gnn_rec":
        gconv_model = GraphSAGEModel(n_hidden,
                                     n_hidden,
                                     n_hidden,
                                     n_layers,
                                     F.relu,
                                     dropout,
                                     aggregator_type)
        model = GNNRec(
            gconv_model=gconv_model,
            input_size=in_feats,
            hidden_size=n_hidden
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Training
        loss_list = []
        pbar = tqdm(range(n_epochs))
        prev_acc = 0
        test_acc_list = []
        for epoch in pbar:
            model.train()
            loss = model(conv_g, loss_g, features, neg_sample_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()))

            acc = RecEvaluate_GnnRec(
                model=model,
                g=conv_g,
                gconv_model=gconv_model,
                features=features,
                users_eval=users_valid,
                movies_eval=movies_valid,
                user_latest_item_dict=user_latest_item,
                dataset=data
            )
            if acc > prev_acc:
                # Save Model
                torch.save(model, model_save_path)
                prev_acc = acc
            pbar.set_description('epoch: %d, loss: %.4f, acc_valid: %.4f' % (epoch, loss.item(), acc))
            loss_list.append(loss.item())
            # Test Dataset
            test_acc = RecEvaluate_GnnRec(
                model=model,
                g=conv_g,
                gconv_model=gconv_model,
                features=features,
                users_eval=users_test,
                movies_eval=movies_test,
                user_latest_item_dict=user_latest_item,
                dataset=data
            )
            test_acc_list.append(test_acc)
    else:
        raise ValueError("The model should be fism or gnn_rec, but {} is given".format(model_name))


    # Print Test HITS
    print('HITS@10:{:.4f}'.format(max(test_acc_list)))

    return loss_list


if __name__ == '__main__':
    # Arguments
    cmd_opt = argparse.ArgumentParser(description='Argparse for running')
    cmd_opt.add_argument('-model', default='fism', help='the model for recommendation',
                         choices=['fism', 'gnn_rec'])
    cmd_opt.add_argument('-sage_hidden_size', default=128, type=int, help='the hidden size of the GraphSAGEModel')
    cmd_opt.add_argument('-sage_layer_num', default=1, type=int, help='the number of layers in the GraphSAGEModel')
    cmd_opt.add_argument('-sage_dropout', default=0.5, type=float, help='the dropout ratio of the GraphSAGEModel')
    cmd_opt.add_argument('-sage_agg_type', default='sum', type=str, help='the aggregator type in the GraphSAGEModel',
                         choices=["sum", "mean", "gcn", "pool", "lstm"])
    cmd_opt.add_argument('-weight_decay', default=0.0005, type=float, help='the weight decay in training')
    cmd_opt.add_argument('-epoch_num', default=100, type=int, help='the number of epoch in training')
    cmd_opt.add_argument('-neg_sample_size', default=40, type=int, help='the number of negative sample size')
    cmd_opt.add_argument('-lr', default=0.001, type=float, help='the learning rate')
    cmd_opt.add_argument('-model_save_path', default="./model.pkl", help='the path for saving model')


    # Parse
    in_feats = features.shape[1]
    cmd_args, _ = cmd_opt.parse_known_args()
    n_hidden = cmd_args.sage_hidden_size
    n_layers = cmd_args.sage_layer_num
    dropout = cmd_args.sage_dropout
    aggregator_type = cmd_args.sage_agg_type
    model_name = cmd_args.model
    weight_decay = cmd_args.weight_decay
    n_epochs = cmd_args.epoch_num
    lr = cmd_args.lr
    neg_sample_size = cmd_args.neg_sample_size
    model_save_path = cmd_args.model_save_path

    a = main_loop(model_name=model_name)




    # Model hyperparameters
    # n_hidden = 128
    # in_feats = features.shape[1]
    # n_layers = 1
    # dropout = 0.5
    # aggregator_type = 'sum'

    # create GraphSAGE model
    # gconv_model = GraphSAGEModel(n_hidden,
    #                              n_hidden,
    #                              n_hidden,
    #                              n_layers,
    #                              F.relu,
    #                              dropout,
    #                              aggregator_type)

    # Training hyperparameters
    # weight_decay = 5e-4
    # n_epochs = 100
    # lr = 1e-3
    # neg_sample_size = 40
    # Model for link prediction
    # model = GNNRec(
    #     gconv_model=gconv_model,
    #     input_size=in_feats,
    #     hidden_size=n_hidden
    # )
    # use optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # initialize graph
    # dur = []
    # prev_acc = 0
    # for epoch in range(n_epochs):
    #     model.train()
    #     loss = model(conv_g, loss_g, features, neg_sample_size)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     print("Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()))
    #
    #     acc = RecEvaluate(
    #         model=model,
    #         g=conv_g,
    #         gconv_model=gconv_model,
    #         features=features,
    #         users_eval=users_valid,
    #         movies_eval=movies_valid,
    #         user_latest_item_dict=user_latest_item,
    #         dataset=data
    #     )
    #     # We have an early stop
    #     # if epoch > 8 and acc <= prev_acc:
    #     #     break
    #     prev_acc = acc
    #
    # # Let's save the trained node embeddings.
    # RecEvaluate(
    #     model=model,
    #     g=conv_g,
    #     gconv_model=gconv_model,
    #     features=features,
    #     users_eval=users_test,
    #     movies_eval=movies_test,
    #     user_latest_item_dict=user_latest_item,
    #     dataset=data
    # )