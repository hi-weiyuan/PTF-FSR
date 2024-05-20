import copy
from parse import args

from models import *
from MoRec import MoRec_Model
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig, AutoModel
from data import read_amazon_data_to_bert_input_form, read_mind_data_to_bert_input_form
from InfoNCE import InfoNCE


class ServerDataset(Dataset):
    def __init__(self, input_seq, valid_seq, neg_seq, seq_len, positions):
        input_seq = ServerDataset.padding_for_listoflist(input_seq)
        valid_seq = ServerDataset.padding_for_listoflist(valid_seq)
        neg_seq = ServerDataset.padding_for_listofarray(neg_seq)
        positions = ServerDataset.padding_for_listoflist(positions)

        self.input_seq = torch.LongTensor(input_seq)
        self.target_seq = torch.LongTensor(valid_seq)
        self.neg_seq = torch.from_numpy(np.array(neg_seq))
        self.pos_seq = torch.LongTensor(positions)
        self.input_len = torch.tensor(seq_len)

    @staticmethod
    def padding_for_listoflist(sequences):
        max_len = max([len(s) for s in sequences])
        sequences = [s + (max_len - len(s)) * [0] for s in sequences]
        return sequences
    @staticmethod
    def padding_for_listofarray(sequences):
        max_len = max([s.shape[0] for s in sequences])
        sequences = [np.pad(s, ((0,max_len - s.shape[0]), (0,0))) for s in sequences]
        return sequences
    @staticmethod
    def padding_for_listoftensor(sequences):
        max_len = max([s.shape[1] for s in sequences])
        sequences = [F.pad(s, (0,0,0,max_len-s.shape[1])) for s in sequences]
        return sequences

    def __len__(self):
        return len(self.input_seq)

    def __getitem__(self, item):
        return self.input_seq[item], self.target_seq[item], self.neg_seq[item], self.pos_seq[item], \
               self.input_len[item], self.valid_label[item], self.neg_label[item]

class ServerDatasetForSSL(ServerDataset):
    def __init__(self, input_seq, valid_seq, neg_seq, seq_len, positions, previous_input_seq, previous_seq_len, previous_positions):
        super(ServerDatasetForSSL, self).__init__(input_seq, valid_seq, neg_seq, seq_len, positions)
        previous_input_seq = ServerDataset.padding_for_listoflist(previous_input_seq)
        previous_positions = ServerDataset.padding_for_listoflist(previous_positions)
        self.previous_input_seq = torch.LongTensor(previous_input_seq)
        self.previous_positions = torch.LongTensor(previous_positions)
        self.previous_seq_len = torch.tensor(previous_seq_len)

    def __getitem__(self, item):
        return self.input_seq[item], self.target_seq[item], self.neg_seq[item], self.pos_seq[item], \
               self.input_len[item], self.previous_input_seq[item], self.previous_seq_len[item], self.previous_positions[item]


class FedRecServer(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.model_type == "GRU4Rec":
            self.model = GRU4Rec(config, config.num_items).to(config.device_id)
        elif config.model_type == "SaSRec":
            self.model = SASRec_Model(config).to(config.device_id)
        elif config.model_type == "MoRec":
            self.tokenizer = BertTokenizer.from_pretrained(config.bert_model_load)
            bert_config = BertConfig.from_pretrained(config.bert_model_load, output_hidden_states=True)
            bert_model = AutoModel.from_pretrained(config.bert_model_load, config=bert_config)
            if "CellPhone" in config.dataset:
                item_to_name, item_content = read_amazon_data_to_bert_input_form(config.path + config.dataset,
                                                                                 "/meta_Cell_Phones_and_Accessories.json.gz",
                                                                                 self.tokenizer, config.num_words_title)
            elif "Baby" in config.dataset:
                item_to_name, item_content = read_amazon_data_to_bert_input_form(config.path + config.dataset,
                                                                                 "/meta_Baby.json.gz", self.tokenizer,
                                                                                 config.num_words_title)
            elif "MIND" in config.dataset:
                item_to_name, item_content = read_mind_data_to_bert_input_form(config.path + config.dataset,
                                                                                 "/mind_items.tsv", self.tokenizer,
                                                                                 config.num_words_title)
            else:
                raise ValueError(f"config.dataset should be in [CellPhone,Baby,MIND]")

            if "small" in config.bert_model_load:
                pooler_para = [69, 70]
                config.word_embedding_dim = 512
            elif "base" in config.bert_model_load:
                pooler_para = [197, 198]
                config.word_embedding_dim = 768
            elif "tiny" in config.bert_model_load:
                pooler_para = [37, 38]
                config.word_embedding_dim = 128
            else:
                raise ValueError(f"config.bert_model_load should be [small, base, tiny]")
            for index, (name, param) in enumerate(bert_model.named_parameters()):
                if index < args.freeze_paras_before or index in pooler_para:
                    param.requires_grad = False
            self.model = MoRec_Model(config, bert_model, item_to_name, item_content).to(config.device_id)
        else:
            raise ImportError(f"the server model type should be in [GRU4Rec,SaSRec, MoRec]")
        self.config = config

        if config.model_type == "MoRec":
            bert_params = []
            recsys_params = []
            for index, (name, param) in enumerate(self.model.named_parameters()):
                if param.requires_grad:
                    if 'bert' in name:
                        bert_params.append(param)
                    else:
                        recsys_params.append(param)
            self.optimizer = torch.optim.AdamW([
                {'params': bert_params, 'lr': config.bert_lr, 'weight_decay': config.l2_weight},
                {'params': recsys_params, 'lr': config.lr, 'weight_decay': config.l2_weight}
            ])
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        self.client_to_sequence = {}
        self.client_previous_sequence = {}
        self.data_for_client = {}
        self.all_items = [i for i in range(1, self.config.num_items)]
        self.neg_num = config.neg_num
        self.tag_infonce = InfoNCE(negative_mode="paired")


    def get_prediction_for_client(self, client, item_embeddings):
        inp_seq = client.query()

        if self.config.model_type == "GRU4Rec":
            inp_len = len(inp_seq)
            pos_seq = list(range(1, inp_len + 1))
            inp_seq = torch.LongTensor(inp_seq).unsqueeze(0).to(self.config.device_id)
            input_len = torch.tensor(inp_len).unsqueeze(0)
            pro = self.model(inp_seq, input_len)
            predictions = pro[:, input_len[0] - 1, :]
        elif self.config.model_type == "SaSRec":
            inp_len = len(inp_seq)
            pos_seq = list(range(1, inp_len + 1))
            inp_seq = torch.LongTensor(inp_seq).unsqueeze(0).to(self.config.device_id)
            input_len = torch.tensor(inp_len).unsqueeze(0)
            pos_seq = torch.LongTensor(pos_seq).unsqueeze(0).to(self.config.device_id)
            seq_emb = self.model(inp_seq, pos_seq)
            last_item_embeddings = seq_emb[:, input_len[0] - 1, :]
            predictions = torch.matmul(last_item_embeddings, self.model.item_emb.transpose(0, 1))
        elif self.config.model_type == "MoRec":
            inp_len = len(inp_seq)
            pos_seq = list(range(1, inp_len + 1))
            inp_seq = torch.LongTensor(inp_seq).unsqueeze(0).to(self.config.device_id)
            input_len = torch.tensor(inp_len).unsqueeze(0)
            pos_seq = torch.LongTensor(pos_seq).unsqueeze(0).to(self.config.device_id)
            seq_emb = self.model.forward_for_prediction(inp_seq, pos_seq, item_embeddings)
            last_item_embeddings = seq_emb[:, input_len[0] - 1, :]
            predictions = torch.matmul(last_item_embeddings, item_embeddings.transpose(0, 1))
        return predictions.squeeze()

    def output_knowledge(self, batch_clients_idx):
        inp_seqs = []
        val_seqs = []
        inp_lens = []
        inp_pos = []
        client_id_to_real_index = {}
        count = 0
        batch_clients_idx = [i+1 for i in batch_clients_idx]
        for i in range(1, self.config.num_users + 1):
            if i not in batch_clients_idx: continue
            client_id_to_real_index[i] = count
            count += 1
            inp_seq_toserver, inp_valid_toserver, len_toserver, pos_toserver = self.client_to_sequence[i]
            inp_seqs.extend(inp_seq_toserver)
            val_seqs.extend(inp_valid_toserver)
            inp_lens.extend(len_toserver)
            inp_pos.extend(pos_toserver)

        inp_seqs = ServerDataset.padding_for_listoflist(inp_seqs)
        val_seqs = ServerDataset.padding_for_listoflist(val_seqs)
        inp_pos = ServerDataset.padding_for_listoflist(inp_pos)

        inp_seqs = torch.LongTensor(inp_seqs).to(self.config.device_id)
        val_seqs = torch.LongTensor(val_seqs).to(self.config.device_id)
        inp_pos = torch.LongTensor(inp_pos).to(self.config.device_id)
        inp_lens = torch.tensor(inp_lens)

        self.model.eval()
        with torch.no_grad():
            if self.config.model_type == "GRU4Rec":
                seq_out, seq_predictions = self.model.forward(inp_seqs, inp_lens, return_seq=True)
            elif self.config.model_type == "SaSRec":
                seq_out = self.model(inp_seqs, inp_pos)
                seq_predictions = torch.matmul(seq_out, self.model.item_emb.transpose(0, 1))
            elif self.config.model_type == "MoRec":
                item_embedding = self.model.update_item_embedding_table()
                seq_out = self.model.forward_for_prediction(inp_seqs, inp_pos, item_embedding)
                seq_predictions = torch.matmul(seq_out, item_embedding.transpose(0, 1))
            else:
                return None
        last_seq_emb = [seq_out[i, last-1, :].view(-1, seq_out.shape[-1]) for i, last in enumerate(inp_lens.tolist())]
        last_seq_emb = torch.cat(last_seq_emb, dim=0)

        similarities = nn.functional.cosine_similarity(last_seq_emb.unsqueeze(1), last_seq_emb, dim=-1)
        diag = torch.diag_embed(torch.diag(similarities))
        similarities = similarities - diag * 2
        _, indexes = torch.topk(similarities, k=1)

        inp_seqs = inp_seqs.cpu()
        val_seqs = val_seqs.cpu()
        inp_pos = inp_pos.cpu()
        inp_lens = inp_lens.tolist()
        seq_predictions = seq_predictions.cpu()
        for client_id in range(1, self.config.num_users + 1):
            if client_id not in batch_clients_idx: continue
            similar_client_indexes = indexes[client_id_to_real_index[client_id]]
            tmp_inp, tmp_val, tmp_len, tmp_pos, tmp_neg, tmp_val_label, tmp_neg_label = [],[],[],[],[],[],[]
            for ids in similar_client_indexes:
                tmp_inp.append(inp_seqs[ids].tolist())
                tmp_val.append(val_seqs[ids].tolist())
                tmp_len.append(inp_lens[ids])
                tmp_pos.append(inp_pos[ids].tolist())
                tmp_val_label.append(torch.gather(seq_predictions[ids], -1, val_seqs[ids].clone().detach().unsqueeze(-1).long()).unsqueeze(0))

                cand = np.setdiff1d(np.array(self.all_items), np.array(inp_seqs[ids].tolist() + val_seqs[ids].tolist()))
                random_neg = np.random.choice(cand, (inp_lens[ids], self.neg_num))
                random_neg =np.pad(random_neg, ((0, seq_predictions[ids].shape[0] - random_neg.shape[0]), (0,0)))
                random_neg_label = torch.gather(seq_predictions[ids], -1, torch.from_numpy(random_neg).long())
                tmp_neg.append(random_neg)
                tmp_neg_label.append(random_neg_label.unsqueeze(0))

            self.data_for_client[client_id] = [tmp_inp, tmp_val, tmp_len, tmp_pos, tmp_neg, tmp_val_label, tmp_neg_label]

    def InfoNCE(self, view1, view2, temperature: float, b_cos: bool = True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

        pos_score = (view1 @ view2.T) / temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()
    def seq_contrastive_calculation(self, previous_inp_seq, previous_inp_poss, inp_len, previous_inp_len, seq_out, item_embeddings=None):
        previous_inp_seq, previous_inp_poss = previous_inp_seq.to(self.config.device_id), previous_inp_poss.to(self.config.device_id)
        if self.config.model_type == "GRU4Rec":
            seq_out_2 = self.model(previous_inp_seq, previous_inp_len)
        elif self.config.model_type == "SaSRec":
            seq_out_2 = self.model(previous_inp_seq, previous_inp_poss)
        elif self.config.model_type == "MoRec":
            seq_out_2 = self.model.forward_for_prediction(previous_inp_seq, previous_inp_poss, item_embeddings)
        last_seq_emb = torch.cat([seq_out[i, last - 1, :].view(-1, seq_out.shape[-1]) for i, last in enumerate(inp_len.tolist())], dim=0)
        last_seq_emb_2 = torch.cat([seq_out_2[i, last - 1, :].view(-1, seq_out_2.shape[-1]) for i, last in
                                    enumerate(previous_inp_len.tolist())], dim=0)
        seq_cl_loss = self.InfoNCE(last_seq_emb, last_seq_emb_2, 0.2)
        return seq_cl_loss
    def target_contrastive_calculation(self, tgt_seq, seq_out, inp_len, item_embeddings=None):
        if item_embeddings is None:
            if self.config.model_type == "SaSRec":
                tgt_emb = self.model.item_emb[tgt_seq[:, -1]].detach().clone()
            else:
                tgt_emb = self.model.item_embedding.weight[tgt_seq[:, -1]].detach().clone()
        else:
            tgt_emb = item_embeddings[tgt_seq[:, -1]].detach().clone()
        similarities = nn.functional.cosine_similarity(tgt_emb.unsqueeze(1), tgt_emb, dim=-1)
        diag = torch.diag_embed(torch.diag(similarities))
        pos_similarities = similarities - diag * 2

        _, pos_indexes = torch.topk(pos_similarities, k=1)
        _, neg_indexes = torch.topk(-similarities, k=int(seq_out.size(0) * 0.1))

        query = torch.cat(
            [seq_out[i, last - 1, :].view(-1, seq_out.shape[-1]) for i, last in enumerate(inp_len.tolist())],
            dim=0)
        pos_key = torch.cat([query[i].view(-1, query.shape[-1]) for i in pos_indexes], dim=0)
        neg_key = torch.stack([query[i] for i in neg_indexes], dim=0)
        tag_loss = self.tag_infonce(query, pos_key, neg_key)
        return tag_loss
    def train_single_batch(self, inp_seq, tgt_seq, neg_seq, pos_seq, inp_len, epoch_id, previous_inp_seq=None, previous_inp_len=None, previous_inp_poss=None):

        inp_seq, tgt_seq, neg_seq, pos_seq = inp_seq.to(self.config.device_id), tgt_seq.to(self.config.device_id), neg_seq.to(self.config.device_id), pos_seq.to(self.config.device_id)

        if self.config.model_type == "GRU4Rec":
            seq_out = self.model(inp_seq, inp_len)
            padding_mask = (torch.not_equal(inp_seq, 0)).unsqueeze(-1).to(self.config.device_id)
            loss = self.model.loss_function(seq_out, padding_mask, tgt_seq, neg_seq, inp_len)
        elif self.config.model_type == "SaSRec":
            seq_out = self.model(inp_seq, pos_seq)
            loss = self.model.loss_function(seq_out, tgt_seq, neg_seq, pos_seq)
        elif self.config.model_type == "MoRec":
            seq_out, item_embeddings = self.model.forward(inp_seq, pos_seq)
            loss = self.model.loss_function(seq_out, tgt_seq, neg_seq, pos_seq, item_embeddings)
            if previous_inp_len is not None:
                seq_cl_loss = self.seq_contrastive_calculation(previous_inp_seq, previous_inp_poss, inp_len, previous_inp_len, seq_out, item_embeddings)
                loss = loss + self.config.pc_cl_control * seq_cl_loss
            if self.config.is_cl and epoch_id > self.config.is_cl_epoch:
                tag_loss = self.target_contrastive_calculation(tgt_seq, seq_out, inp_len, item_embeddings)
                loss = loss + self.config.is_cl_control * tag_loss
        else:
            return None

        if self.config.model_type == "SaSRec" or self.config.model_type == "GRU4Rec":
            if previous_inp_len is not None:
                seq_cl_loss = self.seq_contrastive_calculation(previous_inp_seq, previous_inp_poss, inp_len, previous_inp_len, seq_out)
                loss = loss + self.config.pc_cl_control * seq_cl_loss
            if self.config.is_cl and epoch_id > self.config.is_cl_epoch:
                tag_loss = self.target_contrastive_calculation(tgt_seq, seq_out, inp_len)
                loss = loss + self.config.is_cl_control * tag_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    def train_(self, clients, batch_clients_idx, epoch_id):
        batch_loss = []

        client_inp_seqs = []
        client_neg_seqs = []
        client_valid_seqs = []
        client_seq_lens = []
        client_seq_poses = []
        client_previous_seqs = []
        client_previous_lens = []
        client_previous_poss = []

        for idx in batch_clients_idx:
            client = clients[idx]
            if epoch_id == 1:
                inp_seq_toserver, inp_valid_toserver, inp_neg_toserver, len_toserver, pos_toserver, client_loss = client.train_(None)
            else:
                if client.client_id in self.data_for_client.keys():
                    inp_seq_toserver, inp_valid_toserver, inp_neg_toserver, len_toserver, pos_toserver, client_loss = client.train_(copy.deepcopy(self.data_for_client[client.client_id]))
                else:
                    inp_seq_toserver, inp_valid_toserver, inp_neg_toserver, len_toserver, pos_toserver, client_loss = client.train_(None)

            batch_loss.append(client_loss)

            if client.client_id in self.client_to_sequence.keys():
                self.client_previous_sequence[client.client_id] = copy.deepcopy(self.client_to_sequence[client.client_id])
                client_previous_seqs.extend(self.client_previous_sequence[client.client_id][0])
                client_previous_lens.extend(self.client_previous_sequence[client.client_id][2])
                client_previous_poss.extend(self.client_previous_sequence[client.client_id][3])
            self.client_to_sequence[client.client_id] = \
                [copy.deepcopy(inp_seq_toserver), copy.deepcopy(inp_valid_toserver), copy.deepcopy(len_toserver), copy.deepcopy(pos_toserver)]
            client_inp_seqs.extend(inp_seq_toserver)
            client_neg_seqs.extend(inp_neg_toserver)
            client_valid_seqs.extend(inp_valid_toserver)
            client_seq_lens.extend(len_toserver)
            client_seq_poses.extend(pos_toserver)

        if self.config.pc_cl and epoch_id != 1:
            dataset = ServerDatasetForSSL(client_inp_seqs, client_valid_seqs, client_neg_seqs, client_seq_lens, client_seq_poses,
                                client_previous_seqs, client_previous_lens, client_previous_poss)
        else:
            dataset = ServerDataset(client_inp_seqs, client_valid_seqs, client_neg_seqs, client_seq_lens, client_seq_poses)
        trainloader = DataLoader(dataset, batch_size=self.config.server_bs, shuffle=True)


        server_loss = []
        self.model = self.model.to(self.config.device_id)
        self.model.train()
        for _ in range(self.config.server_epoch):
            server_batch_loss = []
            for _, batch in enumerate(trainloader):
                if self.config.pc_cl and epoch_id != 1:
                    inp_seq, tgt_seq, neg_seq, pos_seq, inp_len, inp_previous_seq, inp_previous_len, inp_previous_pos = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7]
                    loss = self.train_single_batch(inp_seq, tgt_seq, neg_seq, pos_seq, inp_len, epoch_id, inp_previous_seq, inp_previous_len, inp_previous_pos)
                else:
                    inp_seq, tgt_seq, neg_seq, pos_seq, inp_len = batch[0], batch[1], batch[2], batch[3], batch[4]
                    loss = self.train_single_batch(inp_seq, tgt_seq, neg_seq, pos_seq, inp_len, epoch_id)
                server_batch_loss.append(loss)
            server_loss.append(sum(server_batch_loss) / len(server_batch_loss))

        self.output_knowledge(batch_clients_idx)

        return batch_loss, server_loss

    def eval_(self, clients):
        server_test_result_cnt, server_test_result_results = 0, 0.

        self.model = self.model.to(self.config.device_id)
        self.model.eval()
        with torch.no_grad():
            item_embeddings = None
            if self.config.model_type == "MoRec":
                item_embeddings = self.model.update_item_embedding_table()
            for client in clients:
                server_prediction = None
                if client.inp_seq_for_inf is not None:
                    server_prediction = self.get_prediction_for_client(client, item_embeddings)
                server_test_result = client.eval_(server_prediction)

                if server_test_result is not None:
                    server_test_result_cnt += 1
                    server_test_result_results += server_test_result
        return (server_test_result_results / server_test_result_cnt) if server_test_result_cnt!=0 else np.array([0., 0.])