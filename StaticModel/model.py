"""Torch Module for GraphSAGE layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
from torch import nn
from torch.nn import functional as F
import torch
from utils import edge_sampler, score_func, NCE_loss
from dgl import function as fn
import numpy as np

class SAGEConv(nn.Module):
    """
    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    feat_drop : float
        Dropout rate on features, default: ``0``.
    aggregator_type : str
        Aggregator type to use (``sum``, ``mean``, ``gcn``, ``pool``, ``lstm``).
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(in_feats, in_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(in_feats, in_feats, batch_first=True)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(in_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox['m'] # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_feats)),
             m.new_zeros((1, batch_size, self._in_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'neigh': rst.squeeze(0)}

    def forward(self, graph, feat, e_feat):
        r"""Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        graph = graph.local_var()
        feat = self.feat_drop(feat)
        h_self = feat
        graph.edata['e'] = e_feat
        if self._aggre_type == 'sum':
            graph.ndata['h'] = feat
            graph.update_all(fn.u_mul_e('h', 'e', 'm'), fn.sum('m', 'neigh'))
            h_neigh = graph.ndata['neigh']
        elif self._aggre_type == 'mean':
            graph.ndata['h'] = feat
            graph.update_all(fn.u_mul_e('h', 'e', 'm'), fn.mean('m', 'neigh'))
            h_neigh = graph.ndata['neigh']
        elif self._aggre_type == 'gcn':
            graph.ndata['h'] = feat
            graph.update_all(fn.u_mul_e('h', 'e', 'm'), fn.sum('m', 'neigh'))
            # divide in_degrees
            degs = graph.in_degrees().float()
            degs = degs.to(feat.device)
            h_neigh = (graph.ndata['neigh'] + graph.ndata['h']) / (degs.unsqueeze(-1) + 1)
        elif self._aggre_type == 'pool':
            graph.ndata['h'] = F.relu(self.fc_pool(feat))
            graph.update_all(fn.u_mul_e('h', 'e', 'm'), fn.max('m', 'neigh'))
            h_neigh = graph.ndata['neigh']
        elif self._aggre_type == 'lstm':
            graph.ndata['h'] = feat
            graph.update_all(fn.u_mul_e('h', 'e', 'm'), self._lstm_reducer)
            h_neigh = graph.ndata['neigh']
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))
        # GraphSAGE GCN does not require fc_self.
        if self._aggre_type == 'gcn':
            rst = self.fc_neigh(h_neigh)
        else:
            rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self.norm is not None:
            rst = self.norm(rst)
        return rst


class GraphSAGEModel(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_dim,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGEModel, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type,
                                    feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type,
                                        feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, out_dim, aggregator_type,
                                    feat_drop=dropout, activation=None))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h, g.edata['similarity'])
        return h


# Train Item Embeddings
class EncodeLayer(nn.Module):
    def __init__(self, in_feats, num_hidden):
        super(EncodeLayer, self).__init__()
        self.proj = nn.Linear(in_feats, num_hidden)

    def forward(self, feats):
        return self.proj(feats)


class FISM(nn.Module):
    def __init__(self, user_movie_spm, gconv_p, gconv_q, in_feats, num_hidden, beta, gamma):
        super(FISM, self).__init__()
        self.encode = EncodeLayer(in_feats, num_hidden)
        self.num_users = user_movie_spm.shape[0]
        self.num_movies = user_movie_spm.shape[1]
        self.b_u = nn.Parameter(torch.zeros(self.num_users))
        self.b_i = nn.Parameter(torch.zeros(self.num_movies))
        self.user_deg = torch.tensor(user_movie_spm.dot(np.ones(self.num_movies)))
        values = user_movie_spm.data
        indices = np.vstack((user_movie_spm.row, user_movie_spm.col))
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(values)
        self.user_item_spm = torch.sparse_coo_tensor(indices, values, user_movie_spm.shape)
        self.users = user_movie_spm.row
        self.movies = user_movie_spm.col
        self.ratings = user_movie_spm.data
        self.gconv_p = gconv_p
        self.gconv_q = gconv_q
        self.beta = beta
        self.gamma = gamma

    def _est_rating(self, P, Q, user_idx, item_idx):
        bu = self.b_u[user_idx]
        bi = self.b_i[item_idx]
        user_emb = torch.sparse.mm(self.user_item_spm, P)
        user_emb = user_emb[user_idx] / torch.unsqueeze(self.user_deg[user_idx], 1)
        tmp = torch.mul(user_emb, Q[item_idx])
        r_ui = bu + bi + torch.sum(tmp, 1)
        return r_ui

    def est_rating(self, g, features, user_idx, item_idx, neg_item_idx):
        h = self.encode(features)
        P = self.gconv_p(g, h)
        Q = self.gconv_q(g, h)
        r = self._est_rating(P, Q, user_idx, item_idx)
        neg_sample_size = len(neg_item_idx) / len(user_idx)
        neg_r = self._est_rating(P, Q, np.repeat(user_idx, neg_sample_size), neg_item_idx)
        return torch.unsqueeze(r, 1), neg_r.reshape((-1, int(neg_sample_size)))

    def loss(self, P, Q, r_ui, neg_r_ui):
        diff = 1 - (r_ui - neg_r_ui)
        return torch.sum(torch.mul(diff, diff) / 2) \
               + self.beta / 2 * torch.sum(torch.mul(P, P) + torch.mul(Q, Q)) \
               + self.gamma / 2 * (torch.sum(torch.mul(self.b_u, self.b_u)) + torch.sum(torch.mul(self.b_i, self.b_i)))

    def forward(self, g, features, neg_sample_size):
        h = self.encode(features)
        P = self.gconv_p(g, h)
        Q = self.gconv_q(g, h)
        tot = len(self.users)
        pos_idx = np.random.choice(tot, int(tot / 10))
        user_idx = self.users[pos_idx]
        item_idx = self.movies[pos_idx]
        neg_item_idx = np.random.choice(self.num_movies, len(pos_idx) * neg_sample_size)
        r_ui = self._est_rating(P, Q, user_idx, item_idx)
        neg_r_ui = self._est_rating(P, Q, np.repeat(user_idx, neg_sample_size), neg_item_idx)
        r_ui = torch.unsqueeze(r_ui, 1)
        neg_r_ui = neg_r_ui.reshape((-1, int(neg_sample_size)))
        return self.loss(P, Q, r_ui, neg_r_ui)

class GNNRec(nn.Module):
    def __init__(self, gconv_model, input_size, hidden_size):
        super(GNNRec, self).__init__()
        self.encode = EncodeLayer(input_size, hidden_size)
        self.gconv_model = gconv_model

    def forward(self, conv_g, loss_g, features, neg_sample_size):
        emb = self.encode(features)
        emb = self.gconv_model(conv_g, emb)
        pos_g, neg_g = edge_sampler(loss_g, neg_sample_size, return_false_neg=False)
        pos_score = score_func(pos_g, emb)
        neg_score = score_func(neg_g, emb)
        return torch.mean(NCE_loss(pos_score, neg_score, neg_sample_size) * pos_g.edata['weight'])