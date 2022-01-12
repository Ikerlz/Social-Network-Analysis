#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：GNN_Social 
@File    ：data_load.py
@Author  ：Iker Zhe
@Date    ：2022/1/8 15:32 
"""
import pickle
from scipy import sparse as spsp
# loading data
with open("./data/conv_g.pickle", 'rb') as p:
    conv_g = pickle.load(p)
with open("./data/features.pickle", 'rb') as p:
    features = pickle.load(p)
with open("./data/loss_g.pickle", 'rb') as p:
    loss_g = pickle.load(p)
with open("./data/movies_test.pickle", 'rb') as p:
    movies_test = pickle.load(p)
with open("./data/movies_valid.pickle", 'rb') as p:
    movies_valid = pickle.load(p)
with open("./data/users_test.pickle", 'rb') as p:
    users_test = pickle.load(p)
with open("./data/users_valid.pickle", 'rb') as p:
    users_valid = pickle.load(p)
with open("./data/user_latest_item.pickle", 'rb') as p:
    user_latest_item = pickle.load(p)
with open("./data/data.pickle", 'rb') as p:
    data = pickle.load(p)
with open("./data/g.pickle", 'rb') as p:
    g = pickle.load(p)

user_movie_spm = spsp.load_npz('./data/user_movie_spm.npz')
