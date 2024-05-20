import torch
from torch import nn
from torch.nn.init import xavier_normal_, constant_


class User_Encoder(torch.nn.Module):
    def __init__(self, item_num, max_seq_len, item_dim, num_attention_heads, dropout, n_layers):
        super(User_Encoder, self).__init__()
        self.transformer_encoder = TransformerEncoder(n_vocab=item_num, n_position=max_seq_len,
                                                      d_model=item_dim, n_heads=num_attention_heads,
                                                      dropout=dropout, n_layers=n_layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, input_embs, log_mask, local_rank):
        att_mask = (log_mask != 0)
        att_mask = att_mask.unsqueeze(1).unsqueeze(2)
        att_mask = torch.tril(att_mask.expand((-1, -1, log_mask.size(-1), -1))).to(local_rank)
        att_mask = torch.where(att_mask, 0., -1e9)
        return self.transformer_encoder(input_embs, log_mask, att_mask)

class TransformerEncoder(torch.nn.Module):
    def __init__(self, n_vocab, n_position, d_model, n_heads, dropout, n_layers):
        super(TransformerEncoder, self).__init__()
        self.position_embedding = nn.Embedding(n_position, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads, d_inner=d_model * 4, dropout=dropout
                              ) for _ in range(n_layers)])

    def forward(self, input_embs, log_mask, att_mask):
        position_ids = torch.arange(log_mask.size(1), dtype=torch.long, device=log_mask.device)
        position_ids = position_ids.unsqueeze(0).expand_as(log_mask)
        output = self.layer_norm(input_embs + self.position_embedding(position_ids))
        output = self.dropout(output)
        for transformer in self.transformer_blocks:
            output = transformer.forward(output, att_mask)
        return output
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_inner, dropout):

        super().__init__()
        self.multi_head_attention = MultiHeadedAttention(n_heads=n_heads, d_model=d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_inner=d_inner, dropout=dropout)

    def forward(self, block_input, mask):
        output = self.multi_head_attention(block_input, block_input, block_input, mask)
        return self.feed_forward(output)
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.d_v = self.d_k

        self.w_Q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.w_K = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.w_V = nn.Linear(d_model, n_heads * self.d_v, bias=False)
        self.fc = nn.Linear(n_heads * self.d_v, d_model, bias=False)

        self.self_attention = SelfAttention(temperature=self.d_k ** 0.5, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, query, key, value, mask):
        sz_b, len_q, len_k, len_v = query.size(0), query.size(1), key.size(1), value.size(1)
        residual = query

        q = self.w_Q(query).view(sz_b, len_q, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_K(key).view(sz_b, len_k, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_V(value).view(sz_b, len_v, self.n_heads, self.d_v).transpose(1, 2)

        x, attn = self.self_attention(q, k, v, mask=mask)
        x = x.transpose(1, 2).contiguous().view(sz_b, len_q, self.d_model)
        x = self.dropout(self.fc(x))
        return self.layer_norm(residual + x)
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_inner)
        self.w_2 = nn.Linear(d_inner, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.activate = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.dropout(self.w_2(self.activate(self.w_1(x))))
        return self.layer_norm(residual + x)


class SelfAttention(nn.Module):
    def __init__(self, temperature, dropout):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask):
        attn = torch.matmul(query, key.transpose(-2, -1)) / self.temperature
        attn = attn + mask
        p_attn = self.dropout(self.softmax(attn))
        return torch.matmul(p_attn, value), p_attn

class Bert_Encoder(torch.nn.Module):
    def __init__(self, args, bert_model):
        super(Bert_Encoder, self).__init__()
        self.args = args
        self.bert_model = bert_model
        self.fc = nn.Linear(args.word_embedding_dim, args.embed_dim)
        self.activate = nn.GELU()

    def forward(self, text):
        batch_size, num_words = text.shape
        num_words = num_words // 2
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_attmask = torch.narrow(text, 1, num_words, num_words)
        hidden_states = self.bert_model(input_ids=text_ids, attention_mask=text_attmask)[0]
        cls = self.fc(hidden_states[:, 0])
        final_vector = self.activate(cls)
        return final_vector


class MoRec_Model(torch.nn.Module):

    def __init__(self, args, bert_model, item_to_name, item_content):
        super(MoRec_Model, self).__init__()
        self.args = args
        self.item_to_name = item_to_name
        self.item_content = item_content

        self.user_encoder = User_Encoder(args.num_items, args.max_len, args.embed_dim, args.n_heads, args.drop_rate, args.n_blocks)
        self.bert_encoder = Bert_Encoder(args=args, bert_model=bert_model)

        self.criterion = nn.BCEWithLogitsLoss()

    def get_bert_embed(self, sequences):
        sample_item_content = torch.from_numpy(self.item_content[sequences.cpu()])
        sample_item_content = sample_item_content.view(-1, sample_item_content.size(-1))

        if sample_item_content.size(0) <= self.args.num_items:
            sample_item_content = sample_item_content.to(self.args.device_id)
            input_embs_all = self.bert_encoder(sample_item_content)
        else:
            batch_size = int(self.args.num_items / 2)
            data_size = len(sample_item_content)
            num_batch = int(data_size / batch_size)
            tmp_inp_all = []
            for i in range(num_batch):
                tmp_sample = sample_item_content[batch_size * i: (i+1) * batch_size, :]
                tmp_sample = tmp_sample.to(self.args.device_id)
                tmp_inp = self.bert_encoder(tmp_sample)
                tmp_inp_all.append(tmp_inp)
            if num_batch * batch_size < data_size:
                tmp_sample = sample_item_content[num_batch * batch_size:, :]
                tmp_sample = tmp_sample.to(self.args.device_id)
                tmp_inp = self.bert_encoder(tmp_sample)
                tmp_inp_all.append(tmp_inp)
            input_embs_all = torch.cat(tmp_inp_all, dim=0)
        return input_embs_all

    def update_item_embedding_table(self):
        all_items = torch.LongTensor([[i for i in range(0, self.args.num_items)]])
        item_embeddings = self.get_bert_embed(all_items).view(-1, self.args.embed_dim)
        return item_embeddings

    def forward(self, seq, pos):
        item_embeddings = self.update_item_embedding_table()
        input_logs_embs = item_embeddings[seq]
        padding_mask = (torch.not_equal(pos, 0)).to(self.args.device_id)
        seq_emb = self.user_encoder(input_logs_embs, padding_mask, self.args.device_id)
        return seq_emb, item_embeddings
    def forward_for_prediction(self, seq, pos, item_embeddings):
        input_embs = item_embeddings[seq]
        padding_mask = (torch.not_equal(pos, 0)).to(self.args.device_id)
        seq_emb = self.user_encoder(input_embs, padding_mask, self.args.device_id)
        return seq_emb

    def loss_function(self, seq_emb, tgt, neg, pos, item_embeddings):
        target_pos_embs = item_embeddings[tgt]
        neg_emb = item_embeddings[neg]
        pos_score = (seq_emb * target_pos_embs).sum(-1)

        tmp_seq_emb = seq_emb.unsqueeze(dim=2)
        neg_score = (tmp_seq_emb * neg_emb).sum(-1)
        pos_labels, neg_labels = torch.ones(pos_score.shape).to(self.args.device_id), torch.zeros(neg_score.shape).to(self.args.device_id)

        indices = torch.where(pos.cpu() != 0)
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
               self.criterion(neg_score[indices], neg_labels[indices])
        return loss

    def loss_function_with_softlabel(self, seq_emb, tgt, neg, pos, tgt_label, neg_label, item_embeddings):
        target_pos_embs = item_embeddings[tgt]
        neg_emb = item_embeddings[neg]
        pos_score = (seq_emb * target_pos_embs).sum(-1)

        tmp_seq_emb = seq_emb.unsqueeze(dim=2)
        neg_score = (tmp_seq_emb * neg_emb).sum(-1)
        indices = torch.where(pos.cpu() != 0)
        loss = self.criterion(pos_score[indices], tgt_label[indices].squeeze()) + \
               self.criterion(neg_score[indices], neg_label[indices])
        return loss

