import torch
import torch.nn.functional as F
import dgl
import numpy as np
from scipy import stats
def edge_sampler(g, neg_sample_size, edges=None, return_false_neg=True):
    sampler = dgl.contrib.sampling.EdgeSampler(g, batch_size=int(g.number_of_edges() / 10),
                                               seed_edges=edges,
                                               neg_sample_size=neg_sample_size,
                                               negative_mode='tail',
                                               shuffle=True,
                                               return_false_neg=return_false_neg)
    sampler = iter(sampler)
    pos_subg, neg_subg = next(sampler)
    pos_subg.edata['weight'] = g.edata['similarity'][pos_subg.parent_eid]
    return pos_subg, neg_subg


def score_func(g, emb):
    src_nid, dst_nid = g.all_edges(order='eid')
    # Get the node Ids in the parent graph.
    src_nid = g.parent_nid[src_nid]
    dst_nid = g.parent_nid[dst_nid]
    # Read the node embeddings of the source nodes and destination nodes.
    pos_heads = emb[src_nid]
    pos_tails = emb[dst_nid]
    # cosine similarity
    return torch.sum(pos_heads * pos_tails, dim=1)


# NCE loss
def NCE_loss(pos_score, neg_score, neg_sample_size):
    pos_score = F.logsigmoid(pos_score)
    neg_score = F.logsigmoid(-neg_score).reshape(-1, neg_sample_size)
    return -pos_score - torch.sum(neg_score, dim=1)

def RecEvaluate_GnnRec(model, g, gconv_model, features, users_eval, movies_eval, user_latest_item_dict, dataset):
    gconv_model.eval()
    with torch.no_grad():
        emb = model.encode(features)
        emb = model.gconv_model(g, emb)
        hits_10s = []
        # evaluate one user-item interaction at a time
        for u, i in zip(users_eval, movies_eval):
            I_q = user_latest_item_dict[u]
            I = torch.cat([torch.LongTensor([i]), torch.LongTensor(dataset.neg_valid[u])])
            Z_q = emb[I_q]
            Z = emb[I]
            score = (Z_q[None, :] * Z).sum(1).cpu().numpy()
            rank = stats.rankdata(-score, 'min')
            hits_10s.append(rank[0] <= 10)
        # print('HITS@10:{:.4f}'.format(np.mean(hits_10s)))
        return np.mean(hits_10s)

def RecEvaluate_FISM(model, g, features, users_eval, movies_eval, dataset):
    model.eval()
    with torch.no_grad():
        neg_movies_eval = dataset.neg_valid[users_eval].flatten()
        r, neg_r = model.est_rating(g, features, users_eval, movies_eval, neg_movies_eval)
        hits10 = (torch.sum(neg_r > r, 1) <= 10).numpy()
        # print('HITS@10:{:.4f}'.format(np.mean(hits10)))
        return np.mean(hits10)
