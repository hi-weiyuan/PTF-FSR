import copy

from torch.utils.data import DataLoader, Dataset

from models import *
from evaluate import evaluate_recall, evaluate_ndcg

import torch.nn.functional as F
import random


class LocalDataset(Dataset):
    def __init__(self, input_seq, valid_seq, seq_len, neg_seq, positions):
        self.input_seq = torch.LongTensor(input_seq)
        self.target_seq = torch.LongTensor(valid_seq)
        self.pos_seq = torch.LongTensor(positions)
        self.neg_seq = torch.from_numpy(np.array(neg_seq))
        self.input_len = torch.tensor(seq_len)

        self.items = self._flatten_extend()

    def _flatten_extend(self):
        flat_list = []
        flat_list.extend(torch.flatten(self.input_seq).tolist())
        flat_list.extend(torch.flatten(self.target_seq).tolist())
        flat_list.extend(torch.flatten(self.neg_seq).tolist())
        flat_list = list(set(flat_list))
        return flat_list

    def __len__(self):
        return len(self.input_seq)

    def __getitem__(self, item):
        return self.input_seq[item], self.target_seq[item], self.input_len[item], self.neg_seq[item], self.pos_seq[item]


class FedRecClient(nn.Module):
    def __init__(self, config, train_ind, test_ind):
        super().__init__()
        if config.model_type == "GRU4Rec":
            self.model = GRU4Rec(config, config.num_items).to(config.device_id)
        elif config.model_type == "SaSRec":
            self.model = SASRec_Model(config).to(config.device_id)
        else:
            raise ImportError(f"the model type should be in [GRU4Rec, SaSRec...]")
        self.config = config
        self.client_id = self.config.user_id
        self._train_ = train_ind
        self._test_ = test_ind
        self.m_item = config.num_items
        self.all_items = [i for i in range(1, self.m_item)]
        self.neg_num = config.neg_num
        self.max_len = config.max_len

        input_seq, valid_seq, seq_len, neg_seq, pos = self.__construct_local_dataset(train_ind, self.max_len)
        self.input_seq = torch.LongTensor(input_seq).unsqueeze(0)
        self.target_seq = torch.LongTensor(valid_seq).unsqueeze(0)
        self.pos_seq = torch.LongTensor(pos).unsqueeze(0)
        self.input_len = torch.tensor(seq_len).unsqueeze(0)
        self.dataset = LocalDataset([input_seq], [valid_seq], [seq_len], [neg_seq], [pos])
        self.trainloader = DataLoader(self.dataset, batch_size=config.local_bs, shuffle=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.criterion = torch.nn.BCELoss()

        self.trained_item = copy.deepcopy(self.dataset.items)

        self.replace_ratio = config.replace_ratio

        self.inp_seq_for_inf = None
        self.inp_val_for_inf = None

    def update_trained_item(self, tensor):
        self.trained_item.extend(torch.flatten(tensor).tolist())
        self.trained_item = list(set(self.trained_item))

    def __construct_local_dataset(self, train_ind, max_len):
        start = len(train_ind) > max_len and -max_len or 0
        end = len(train_ind) > max_len and max_len - 1 or len(train_ind) - 1
        input_seq = train_ind[start:-1]
        valid_seq = train_ind[start+1:]

        seq_len = end
        pos = list(range(1, end + 1))
        all_items = [i for i in range(1, self.m_item)]
        cand = np.setdiff1d(np.array(all_items), np.array(train_ind))
        neg_seq = np.random.choice(cand, (seq_len, self.neg_num))
        return input_seq, valid_seq, seq_len, neg_seq, pos

    def train_single_batch(self, inp_seq, target_seq, neg_seq, pos_seq, inp_len):
        inp_seq, target_seq, neg_seq, pos_seq = inp_seq.to(self.config.device_id), target_seq.to(self.config.device_id), neg_seq.to(self.config.device_id), pos_seq.to(self.config.device_id)
        if self.config.model_type == "GRU4Rec":
            seq_out = self.model(inp_seq, inp_len)
            padding_mask = (torch.not_equal(inp_seq, 0)).to(self.config.device_id)
            loss = self.model.loss_function(seq_out, padding_mask, target_seq, neg_seq, inp_len)
        elif self.config.model_type == "SaSRec":
            seq_out = self.model(inp_seq, pos_seq)
            loss = self.model.loss_function(seq_out, target_seq, neg_seq, pos_seq)
        else:
            return None

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss
    def train_single_batch_with_softlabel(self, inp_seq, target_seq, neg_seq, pos_seq, inp_len, val_label, neg_label):
        inp_seq, tgt_seq, neg_seq, pos_seq = inp_seq.to(self.config.device_id), target_seq.to(
            self.config.device_id), neg_seq.to(self.config.device_id), pos_seq.to(self.config.device_id)
        tgt_label, neg_label = val_label.to(self.config.device_id), neg_label.to(self.config.device_id)
        if self.config.model_type == "GRU4Rec":
            seq_out = self.model(inp_seq, inp_len)
            padding_mask = (torch.not_equal(inp_seq, 0)).to(self.config.device_id)
            loss = self.model.loss_function_with_softlabel(seq_out, padding_mask, tgt_seq, neg_seq, tgt_label, neg_label)
        elif self.config.model_type == "SaSRec":
            seq_out = self.model(inp_seq, pos_seq)
            loss = self.model.loss_function_with_softlabel(seq_out, tgt_seq, neg_seq, pos_seq, tgt_label, neg_label)
        else:
            return None
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    def train_(self, server_data):
        if server_data is not None:
            server_inp, server_val, server_len, server_pos, server_neg, server_val_label, server_neg_label = server_data
            server_inp = torch.LongTensor(server_inp)
            self.update_trained_item(server_inp)
            server_val = torch.LongTensor(server_val)
            self.update_trained_item(server_val)
            server_len = torch.tensor(server_len)
            server_pos = torch.LongTensor(server_pos)
            server_neg = torch.from_numpy(np.array(server_neg))
            self.update_trained_item(server_neg)
            server_val_label = torch.cat(server_val_label, dim=0)
            server_neg_label = torch.cat(server_neg_label, dim=0)

        epoch_loss = []
        self.model = self.model.to(self.config.device_id)
        self.model.train()
        for _ in range(self.config.local_epoch):
            batch_loss = []
            for batch_id, batch in enumerate(self.trainloader):
                input_seq, valid_seq, seq_len, neg_seq, positions = batch[0], batch[1], batch[2], batch[3], batch[4]
                loss = self.train_single_batch(input_seq, valid_seq, neg_seq, positions, seq_len)
                batch_loss.append(loss)
            if server_data is not None:
                loss = self.train_single_batch_with_softlabel(server_inp, server_val, server_neg, server_pos, server_len, server_val_label, server_neg_label)
                batch_loss.append(loss)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        inp_seq_toserver, inp_valid_toserver, inp_neg_toserver, len_toserver, pos_toserver = self.transfer_to_server()
        self.model = self.model.cpu()
        self.inp_seq_for_inf = copy.deepcopy(inp_seq_toserver)
        self.inp_val_for_inf = copy.deepcopy(inp_valid_toserver)

        return inp_seq_toserver, inp_valid_toserver, inp_neg_toserver, len_toserver, pos_toserver, sum(epoch_loss) / len(epoch_loss)

    def call_model_for_forward(self, a, b, c):
        if self.config.model_type == "GRU4Rec":
            inp_seq, inp_len = a.to(self.config.device_id), b
            seq_emb = self.model(inp_seq, inp_len)
            predictions_for_sequence = torch.matmul(seq_emb, self.model.item_embedding.weight.transpose(0, 1))
        elif self.config.model_type == "SaSRec":
            inp_seq, pos_seq = a.to(self.config.device_id), c.to(self.config.device_id)
            seq_emb = self.model(inp_seq, pos_seq)
            predictions_for_sequence = torch.matmul(seq_emb, self.model.item_emb.transpose(0, 1))
        else:
            raise ValueError(f"model type should be in [GRU4Rec, SaSRec]")
        return predictions_for_sequence

    def transfer_to_server(self):
        self.model.eval()
        with torch.no_grad():
            trained_items = self.trained_item
            predictions_for_sequence = self.call_model_for_forward(self.input_seq, self.input_len, self.pos_seq)
            predictions_for_sequence = predictions_for_sequence.squeeze()
            count = 10
            original_seq = self.input_seq.squeeze().tolist()
            original_valid = self.target_seq.squeeze().tolist()
            sequences = [[original_seq[0]] for _ in range(count)]

            for i in range(1, self.input_len[0]):
                if random.random() < self.replace_ratio:
                    prediction_at_i = predictions_for_sequence[i, trained_items]

                    prob_i = F.softmax(prediction_at_i * self.config.epsilon).cpu().numpy()
                    prob_i /= prob_i.sum()

                    rep = np.random.choice(trained_items, size=count, replace=True, p=prob_i).tolist()
                    sequences = [seq + [rep[ii]] for ii, seq in enumerate(sequences)]
                else:
                    sequences = [seq + [original_seq[i]] for seq in sequences]
            cand_inp = torch.LongTensor(sequences)
            cand_input_len = torch.tensor([self.input_len[0]] * len(cand_inp))
            cand_pos_seq = torch.LongTensor([list(range(1, cand_input_len[0] + 1))] * len(cand_inp))
            cand_predictions = self.call_model_for_forward(cand_inp, cand_input_len, cand_pos_seq)
            last_prediction = cand_predictions[:, cand_input_len[0] -1, :]

            last_prediction = last_prediction.squeeze()
            last_prediction = last_prediction[:, self._train_[-1]].squeeze().cpu()
            tmp, sequence_k = torch.topk(last_prediction, 1)
            sequence_k = sequence_k.cpu().tolist()

            final_input_sequences = [sequences[i] for i in sequence_k]
            final_neg_sequences = []
            for i in range(1):
                cand = np.setdiff1d(np.array(trained_items), np.array(final_input_sequences[i]))
                neg_seq = np.random.choice(cand, (cand_input_len[0], self.neg_num))
                final_neg_sequences.append(neg_seq)
            final_valid_sequences = [seq[1:] + [original_valid[-1]] for seq in final_input_sequences]

            return final_input_sequences, final_valid_sequences, final_neg_sequences, [self.input_len[0].item()] * len(final_input_sequences), [list(range(1, cand_input_len[0] + 1))] * len(final_input_sequences)

    def query(self):
        tmp = self.inp_seq_for_inf[0] + self.inp_val_for_inf[0][-1:]
        return tmp[-self.max_len:]

    def eval_(self, server_prediction):
        server_test_result = None
        if server_prediction is not None:
            server_prediction[self._train_ + [0]] = - (1 << 10)
            if self._test_:
                hr_at_20 = evaluate_recall(server_prediction, self._test_, 20)
                ndcg_at_20 = evaluate_ndcg(server_prediction, self._test_, 20)
                server_test_result = np.array([hr_at_20, ndcg_at_20])
            else:
                server_test_result = None

        return server_test_result

