from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np
import random
import torch.optim as optim

from util.cmd_args import cmd_args
from util.consts import DEVICE
from util.dataset import train_data, test_data
from util.bipartite_graph import bg
from util.recorder import cur_time, dur_dist
from deepcoevolve import DeepCoevolve
from tqdm import tqdm


def load_data():
    train_data.load_events(cmd_args.train_file, 'train')
    test_data.load_events(cmd_args.test_file, 'test')

    for e_idx, cur_event in enumerate(test_data.ordered_events):
        cur_event.global_idx += train_data.num_events
        if cur_event.prev_user_event is None:
            continue
        train_events = train_data.user_event_lists[cur_event.user]
        if len(train_events) == 0:
            continue
        assert train_events[-1].t <= cur_event.t
        cur_event.prev_user_event = train_events[-1]
        cur_event.prev_user_event.next_user_event = cur_event

    print('# train:', train_data.num_events, '# test:', test_data.num_events)
    print('totally', cmd_args.num_users, 'users,', cmd_args.num_items, 'items')


def main_loop():
    bg.reset()
    for event in train_data.ordered_events:
        bg.add_event(event.user, event.item)

    model = DeepCoevolve(cmd_args.num_users, cmd_args.num_items, cmd_args.embed_dim,
                         score_func=cmd_args.score_func, dt_type=cmd_args.dt_type, max_norm=cmd_args.max_norm,
                         pp_model=cmd_args.pp_model).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate)

    cur_time.reset(0)
    for e in train_data.ordered_events:
        cur_time.update_event(e.user, e.item, e.t)
    rc_dump = cur_time.dump()
    mar_list = []
    mae_list = []
    rmse_list = []
    for epoch in range(cmd_args.num_epochs):

        cur_time.load_dump(*rc_dump)
        mar, mae, rmse = model(train_data.ordered_events[-1].t,
                               test_data.ordered_events[:5000],
                               phase='test')
        print(mar, mae, rmse)
        mar_list.append(mar)
        mae_list.append(mae)
        rmse_list.append(rmse)
        pbar = tqdm(range(cmd_args.iters_per_val))
        for it in pbar:
            cur_pos = np.random.randint(train_data.num_events - cmd_args.bptt)

            T_begin = 0
            if cur_pos:
                T_begin = train_data.ordered_events[cur_pos - 1].t

            event_mini_batch = train_data.ordered_events[cur_pos:cur_pos + cmd_args.bptt]

            optimizer.zero_grad()
            loss, mae, rmse = model(T_begin, event_mini_batch, phase='train')
            pbar.set_description('epoch: %.2f, loss: %.4f, mae: %.4f, rmse: %.4f' % (
            epoch + (it + 1) / len(pbar), loss.item(), mae, rmse))

            loss.backward()

            if cmd_args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)

            optimizer.step()
            model.normalize()
        dur_dist.print_dist()


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    load_data()

    main_loop()
