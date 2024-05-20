import torch
import torch.nn as nn
import numpy as np

class GRU4Rec(nn.Module):
    def __init__(self, args, item_maxid):
        super().__init__()
        self.embed_dim = args.embed_dim
        self.device = args.device_id
        self.config = args
        if args.embed_dim != -1:
            self.item_embedding = nn.Embedding(item_maxid, args.embed_dim, padding_idx=0)
            self.gru_layer = nn.GRU(
                input_size=args.embed_dim,
                hidden_size=args.hidden_size,
                batch_first=True)
        else:
            self.item_one_hot_embedding = torch.eye(item_maxid, dtype=torch.float32)
            self.gru_layer = nn.GRU(
                input_size=item_maxid + 1,
                hidden_size=args.hidden_size,
                batch_first=True)
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.output_layer = nn.Linear(args.hidden_size, args.embed_dim, bias=False)
        self.last_layer_norm = torch.nn.LayerNorm(self.embed_dim, eps=1e-8)
        self.rec_loss = torch.nn.BCEWithLogitsLoss()
        self.init_param()

    def init_param(self):
        if self.embed_dim != -1:
            nn.init.kaiming_normal_(self.item_embedding.weight, mode='fan_out')
        nn.init.orthogonal_(self.gru_layer.weight_ih_l0)
        nn.init.orthogonal_(self.gru_layer.weight_hh_l0)
        nn.init.constant_(self.gru_layer.bias_ih_l0, 0.0)
        nn.init.constant_(self.gru_layer.bias_hh_l0, 0.0)
        nn.init.kaiming_normal_(self.output_layer.weight, mode='fan_out')

    def forward(self, x, x_lens):
        max_len = x.shape[1]
        if self.embed_dim != -1:
            seq_embedding = self.item_embedding(x)
        else:
            seq_embedding = self.item_one_hot_embedding[x.long().flatten()]
            seq_embedding = seq_embedding.view(x.shape[0], x.shape[1], -1)
        seq_embedding = torch.nn.utils.rnn.pack_padded_sequence(seq_embedding, x_lens, batch_first=True,
                                                                enforce_sorted=False)
        gru_out, _ = self.gru_layer(seq_embedding)
        gru_out, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True, total_length=max_len)

        seq_out = self.output_layer(gru_out)
        seq_out = self.last_layer_norm(seq_out)
        return seq_out

    def loss_function(self, seq_out, padding_mask, target, neg, seq_len):
        y_emb = self.item_embedding(target)
        neg_emb = self.item_embedding(neg)
        pos_logits = (seq_out * y_emb).sum(dim=-1)
        tmp_seq_emb = seq_out.unsqueeze(dim=2)
        neg_logits = (tmp_seq_emb * neg_emb).sum(dim=-1)
        pos_labels, neg_labels = torch.ones(pos_logits.shape).to(self.config.device_id), torch.zeros(neg_logits.shape).to(self.config.device_id)
        indices = np.where(padding_mask.cpu() !=0)
        loss = self.rec_loss(pos_logits[indices], pos_labels[indices])
        loss += self.rec_loss(neg_logits[indices], neg_labels[indices])
        return loss

    def loss_function_with_softlabel(self, seq_out, padding_mask, target, neg, tgt_label, neg_label):
        y_emb = self.item_embedding(target)
        neg_emb = self.item_embedding(neg)
        pos_logits = (seq_out * y_emb).sum(dim=-1)
        tmp_seq_emb = seq_out.unsqueeze(dim=2)
        neg_logits = (tmp_seq_emb * neg_emb).sum(dim=-1)
        indices = np.where(padding_mask.cpu() != 0)
        loss = self.rec_loss(pos_logits[indices], tgt_label[indices].squeeze(-1))
        loss += self.rec_loss(neg_logits[indices], neg_label[indices])
        return loss

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate,activation='relu'):
        super(PointWiseFeedForward, self).__init__()
        act = None
        if activation == 'relu':
            act = torch.nn.ReLU()
        elif activation == 'gelu':
            act = torch.nn.GELU()
        self.pwff = torch.nn.Sequential(
            torch.nn.Linear(hidden_units, hidden_units),
            act,
            torch.nn.Linear(hidden_units, hidden_units),
            torch.nn.Dropout(p=dropout_rate)
        )

    def forward(self, inputs):
        outputs = self.pwff(inputs)
        outputs = outputs + inputs
        return outputs


class SASRec_Model(nn.Module):
    def __init__(self, args):
        super(SASRec_Model, self).__init__()
        self.config = args
        self.item_maxid = args.num_items
        self.emb_size = args.embed_dim
        self.block_num = args.n_blocks
        self.head_num = args.n_heads
        self.drop_rate = args.drop_rate
        self.max_len = args.max_len
        self.rec_loss = torch.nn.BCEWithLogitsLoss()
        self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        self.item_emb = nn.Parameter(initializer(torch.empty(self.item_maxid, self.emb_size)))
        self.pos_emb = nn.Parameter(initializer(torch.empty(self.max_len+1, self.emb_size)))
        self.attention_layer_norms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layer_norms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.emb_dropout = torch.nn.Dropout(self.drop_rate)
        self.last_layer_norm = torch.nn.LayerNorm(self.emb_size, eps=1e-8)

        for n in range(self.block_num):
            self.attention_layer_norms.append(torch.nn.LayerNorm(self.emb_size, eps=1e-8))
            new_attn_layer =  torch.nn.MultiheadAttention(self.emb_size, self.head_num, self.drop_rate)
            self.attention_layers.append(new_attn_layer)
            self.forward_layer_norms.append(torch.nn.LayerNorm(self.emb_size, eps=1e-8))
            new_fwd_layer = PointWiseFeedForward(self.emb_size, self.drop_rate)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, seq, pos, seq_emb=None):
        if seq_emb is None:
            seq_emb = self.item_emb[seq]
            seq_emb *= self.emb_size ** 0.5
        pos_emb = self.pos_emb[pos]
        seq_emb = seq_emb + pos_emb
        seq_emb = self.emb_dropout(seq_emb)

        timeline_mask = torch.BoolTensor(pos.cpu() == 0).to(self.config.device_id)
        seq_emb *= ~timeline_mask.unsqueeze(-1)
        tl = seq_emb.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool).to(self.config.device_id))
        for i in range(len(self.attention_layers)):
            seq_emb = torch.transpose(seq_emb, 0, 1)
            normalized_emb = self.attention_layer_norms[i](seq_emb)
            mha_outputs, _ = self.attention_layers[i](normalized_emb, seq_emb, seq_emb, attn_mask=attention_mask)
            seq_emb = normalized_emb + mha_outputs
            seq_emb = torch.transpose(seq_emb, 0, 1)
            seq_emb = self.forward_layer_norms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)
            seq_emb *=  ~timeline_mask.unsqueeze(-1)
        seq_emb = self.last_layer_norm(seq_emb)
        return seq_emb

    def loss_function(self, seq_emb, y, neg, pos):
        y_emb = self.item_emb[y]
        neg_emb = self.item_emb[neg]
        pos_logits = (seq_emb * y_emb).sum(dim=-1)
        tmp_seq_emb = seq_emb.unsqueeze(dim=2)
        neg_logits = (tmp_seq_emb * neg_emb).sum(dim=-1)
        pos_labels, neg_labels = torch.ones(pos_logits.shape).to(self.config.device_id), torch.zeros(neg_logits.shape).to(self.config.device_id)
        indices = np.where(pos.cpu() != 0)
        loss = self.rec_loss(pos_logits[indices], pos_labels[indices])
        loss += self.rec_loss(neg_logits[indices], neg_labels[indices])
        return loss

    def loss_function_with_softlabel(self, seq_emb, y, neg, pos, tgt_label, neg_label):
        y_emb = self.item_emb[y]
        neg_emb = self.item_emb[neg]
        pos_logits = (seq_emb * y_emb).sum(dim=-1)
        tmp_seq_emb = seq_emb.unsqueeze(dim=2)
        neg_logits = (tmp_seq_emb * neg_emb).sum(dim=-1)
        indices = np.where(pos.cpu() != 0)
        loss = self.rec_loss(pos_logits[indices], tgt_label[indices].squeeze())
        loss += self.rec_loss(neg_logits[indices], neg_label[indices])
        return loss
