from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from util.cmd_args import cmd_args

class Event(object):
    def __init__(self, user, item, t, phase):
        self.user = user
        self.item = item
        self.t = t
        self.phase = phase

        self.next_user_event = None
        self.prev_user_event = None
        self.prev_item_event = None
        self.global_idx = None


class Dataset(object):
    def __init__(self):
        self.user_event_lists = []
        self.item_event_lists = []
        self.ordered_events = []
        self.num_events = 0

    def load_events(self, filename, phase):
        self.user_event_lists = [[] for _ in range(cmd_args.num_users)]
        self.item_event_lists = [[] for _ in range(cmd_args.num_items)]

        with open(filename, 'r') as f:
            rows = f.readlines()
            for row in rows:
                user, item, t = row.split()[:3]
                user = int(user)
                item = int(item)
                t = float(t) * cmd_args.time_scale
                cur_event = Event(user, item, t, phase)
                self.ordered_events.append(cur_event)
        
        self.ordered_events = sorted(self.ordered_events, key=lambda x: x.t)
        for i in range(len(self.ordered_events)):
            cur_event = self.ordered_events[i]

            cur_event.global_idx = i
            user = cur_event.user
            item = cur_event.item
            
            if len(self.user_event_lists[user]):
                cur_event.prev_user_event = self.user_event_lists[user][-1]
            if len(self.item_event_lists[item]):
                cur_event.prev_item_event = self.item_event_lists[item][-1]
            if cur_event.prev_user_event is not None:
                cur_event.prev_user_event.next_user_event = cur_event
            self.user_event_lists[user].append(cur_event)
            self.item_event_lists[item].append(cur_event)

        self.num_events = len(self.ordered_events)

    def clear(self):
        self.user_event_lists = []
        self.item_event_lists = []
        self.ordered_events = []

train_data = Dataset()
test_data = Dataset()
