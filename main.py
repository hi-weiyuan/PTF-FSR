import gc

import torch
import random
import numpy as np
from time import time
from parse import args
from data import load_dataset
from client import FedRecClient
from server import FedRecServer
from tqdm import tqdm

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

class ClientConfig(object):
    def __init__(self, user_id, num_items, latent_dim, device_id,
                 local_epoch, local_lr, local_bs, model_type, num_neg, max_len,
                 n_blocks, n_heads, replace_ratio,
                 client_emb_dim=-1, dropout = 0.5, epsilon=1.):
        self.user_id = user_id
        self.num_items = num_items
        self.hidden_size = latent_dim
        self.device_id = device_id
        self.model_type = model_type
        self.embed_dim = client_emb_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.drop_rate = dropout

        self.local_epoch = local_epoch
        self.lr = local_lr
        self.local_bs = local_bs
        self.neg_num = num_neg
        self.max_len = max_len

        self.replace_ratio = replace_ratio
        self.epsilon = epsilon


class ServerConfig(object):
    def __init__(self, num_users, num_items, latent_dim, model_type, device_id,
                 global_lr, server_bs, server_epoch, num_neg, max_len,
                 n_blocks, n_heads,
                 server_emb_dim=-1, dropout=0.5, l2_weight=0.01, bert_lr=5e-5, path="", dataset="", bert_model_load="",
                 freeze_paras_before=0, num_words_title=30, pc_cl=True, is_cl=True,
                 pc_cl_control=0.5, is_cl_control=0.5, is_cl_epoch=5):
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_size = latent_dim

        self.model_type = model_type
        self.device_id = device_id
        self.embed_dim = server_emb_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.drop_rate = dropout

        self.lr = global_lr
        self.l2_weight = l2_weight
        self.bert_lr = bert_lr
        self.server_bs = server_bs
        self.server_epoch = server_epoch

        self.neg_num = num_neg
        self.max_len = max_len
        self.path = path
        self.dataset = dataset
        self.bert_model_load = bert_model_load
        self.freeze_paras_before = freeze_paras_before
        self.num_words_title = num_words_title

        self.pc_cl = pc_cl
        self.is_cl = is_cl
        self.pc_cl_control = pc_cl_control
        self.is_cl_control = is_cl_control
        self.is_cl_epoch = is_cl_epoch




def main():
    args_str = ",".join([("%s=%s" % (k, v)) for k, v in args.__dict__.items()])
    print("Arguments: %s " % args_str)

    t0 = time()
    m_item, all_train_ind, all_test_ind, items_popularity, all_user = load_dataset(args.path + args.dataset)

    server_config = ServerConfig(len(all_user), m_item, args.server_dim,
                                 args.server_model_type, args.server_device,
                                 args.global_lr, args.server_bs,
                                 args.server_epoch, args.num_neg, args.max_len,
                                 args.server_n_blocks, args.server_n_heads,
                                 args.server_emb_dim, args.server_dropout, args.l2_weight, args.bert_lr, args.path, args.dataset,
                                 args.bert_model_load, args.freeze_paras_before, args.num_words_title,
                                 args.pc_cl, args.is_cl, args.pc_cl_control, args.is_cl_control, args.is_cl_epoch)
    server = FedRecServer(server_config)
    clients = []
    for user_id, train_ind, test_ind in tqdm(zip(all_user, all_train_ind, all_test_ind)):
        client_config = ClientConfig(user_id, m_item, args.client_dim, args.client_device,
                                     args.local_epoch, args.local_lr, args.local_bs,
                                     args.local_model_type, args.num_neg,
                                     args.max_len, args.client_n_blocks, args.client_n_heads, args.replace_ratio,
                                     args.client_emb_dim, args.dropout, args.epsilon)
        clients.append(FedRecClient(client_config, train_ind, test_ind))

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" %
          (time() - t0, len(clients), m_item,
           sum([len(i) for i in all_train_ind]),
           sum([len(i) for i in all_test_ind])))
    print("output format: (server: {Recall@20, NDCG@20}), {client: Recall@20, NDCG@20}), ensemble: {Recall@20, NDCG@20}))")

    t1 = time()
    server_result = server.eval_(clients)
    print("Iteration 0(init)" +
          ", result: (%.7f, %.7f)" % tuple(server_result) +
          " [%.1fs]" % (time() - t1))

    for epoch in range(1, args.epochs + 1):
        t1 = time()
        rand_clients = np.arange(len(clients))
        np.random.shuffle(rand_clients)

        client_losses = []
        server_losses = []

        for i in range(0, len(rand_clients), args.batch_size):
            batch_clients_idx = rand_clients[i: i + args.batch_size]
            client_loss, server_loss = server.train_(clients, batch_clients_idx, epoch)
            client_losses.extend(client_loss)
            server_losses.extend(server_loss)

        t2 = time()
        server_result = server.eval_(clients)
        print("Iteration %d, client loss = %.5f, server loss = %.5f [%.1fs]" % (epoch,  sum(client_losses) / len(client_losses), sum(server_losses) / len(server_losses), t2 - t1) +
              ", result: (%.7f, %.7f)" % tuple(server_result) +
              " [%.1fs]" % (time() - t2))

