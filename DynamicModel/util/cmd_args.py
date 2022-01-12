from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import os
import pickle as cp

cmd_opt = argparse.ArgumentParser(description='Argparser for coevolve')
cmd_opt.add_argument('-save_dir', default='.', help='result output root')
cmd_opt.add_argument('-dropbox', default=None, help='dropbox folder')
cmd_opt.add_argument('-init_model_dump', default=None, help='model dump')
cmd_opt.add_argument('-data_name', default=None, help='dataset name')
cmd_opt.add_argument('-phase', default=None, help='phase')
cmd_opt.add_argument('-dt_type', default='last', help='last/cur')
cmd_opt.add_argument('-int_act', default='exp', help='activation function for intensity', choices=['exp', 'softplus'])
cmd_opt.add_argument('-score_func', default='log_ll', help='log_ll/comp/intensity')

cmd_opt.add_argument('-is_training', default=True, type=eval, help='is training')

cmd_opt.add_argument('-meta_file', default=None, help='meta_file')
cmd_opt.add_argument('-train_file', default=None, help='train_file')
cmd_opt.add_argument('-test_file', default=None, help='test_file')

cmd_opt.add_argument('-embed_dim', default=128, type=int, help='embedding dim of gnn')
cmd_opt.add_argument('-bptt', default=100, type=int, help='bptt size')

cmd_opt.add_argument('-num_items', default=0, type=int, help='num items')
cmd_opt.add_argument('-num_users', default=0, type=int, help='num users')
cmd_opt.add_argument('-neg_items', default=100, type=int, help='neg items')
cmd_opt.add_argument('-neg_users', default=100, type=int, help='neg users')

cmd_opt.add_argument('-max_norm', default=None, type=float, help='max embed norm')
cmd_opt.add_argument('-time_scale', default=1.0, type=float, help='time scale')
cmd_opt.add_argument('-time_lb', default=0.1, type=float, help='min time dur')
cmd_opt.add_argument('-time_ub', default=0.1, type=float, help='max time dur')

cmd_opt.add_argument('-seed', default=19260817, type=int, help='seed')

cmd_opt.add_argument('-learning_rate', default=1e-3, type=float, help='learning rate')
cmd_opt.add_argument('-grad_clip', default=5, type=float, help='clip gradient')
cmd_opt.add_argument('-num_epochs', default=10000, type=int, help='number of training epochs')
cmd_opt.add_argument('-iters_per_val', default=100, type=int, help='number of iterations per evaluation')
cmd_opt.add_argument('-batch_size', default=64, type=int, help='batch size for training')
cmd_opt.add_argument('-pp_model', default='reyleigh', type=str, help='the point process model, reyleigh or hawkes')

cmd_args, _ = cmd_opt.parse_known_args()

if cmd_args.save_dir is not None:
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)

assert cmd_args.meta_file is not None
with open(cmd_args.meta_file, 'r') as f:
    row = f.readline()
    row = [int(t) for t in row.split()[:2]]
    cmd_args.num_users, cmd_args.num_items = row

print(cmd_args)
