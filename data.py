import numpy as np
import pickle
import gzip

def load_file(file_path):
    m_item, all_pos = 0, []
    all_user = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            pos = list(map(int, line.rstrip().split(' ')))[1:]
            all_user.append(list(map(int, line.rstrip().split(' ')))[0])
            if pos:
                m_item = max(m_item, max(pos)+1)
            all_pos.append(pos)

    return m_item, all_pos, all_user


def load_dataset(path):
    m_item = 0
    m_item_, all_train_ind, m_user = load_file(path + "/train.dat")
    m_item = max(m_item, m_item_)
    m_item_, all_test_ind, m_user_ = load_file(path + "/test.dat")
    m_item = max(m_item, m_item_)
    assert m_user == m_user_, f"m_user != m_user_: {m_user} !=\n {m_user_}"

    items_popularity = np.zeros(m_item)
    for items in all_train_ind:
        for item in items:
            items_popularity[item] += 1
    for items in all_test_ind:
        for item in items:
            items_popularity[item] += 1

    return m_item, all_train_ind, all_test_ind, items_popularity, m_user

def read_amazon_data_to_bert_input_form(path, json_gz_file_name, bert_tokenizer, num_words_title):
    def parse(path):
        g = gzip.open(path, 'r')
        for l in g:
            yield eval(l)
    with open(path + "/item_to_name.pkl", "rb") as f:
        item_to_name = pickle.load(f)

    metadata = {}
    for data in parse(path + json_gz_file_name):
        item_name = data["asin"]
        metadata[item_name] = data

    item_id_to_bert_tokenized = {}
    no_title_count = 0
    even_no_description = 0
    for item_id, item_name in item_to_name.items():
        if "title" in metadata[item_name].keys() and len(metadata[item_name]["title"].strip()) > 0:
            title = metadata[item_name]["title"]
        else:
            no_title_count += 1
            if "description" in metadata[item_name].keys() and len(metadata[item_name]["description"].strip()) > 0:
                title = metadata[item_name]["description"]
            else:
                even_no_description += 1
                title = " ".join(metadata[item_name]["categories"][0])
                if len(title.strip()) == 0:
                    raise ValueError(f"there is one odd item: {metadata[item_name]}")
        item_id_to_bert_tokenized[item_id] = bert_tokenizer(title.lower(), max_length=num_words_title, padding='max_length', truncation=True)

    max_item_id = max([item_id for item_id in item_to_name.keys()])
    title_input_ids = np.zeros((max_item_id + 1, num_words_title), dtype='int32')
    title_attn_mask = np.zeros((max_item_id + 1, num_words_title), dtype='int32')
    for item_id, title_tokenized in item_id_to_bert_tokenized.items():
        title_input_ids[item_id] = title_tokenized["input_ids"]
        title_attn_mask[item_id] = title_tokenized['attention_mask']

    item_content = np.concatenate([title_input_ids, title_attn_mask], axis=1)

    return item_to_name, item_content


def read_mind_data_to_bert_input_form(path, item_file, bert_tokenizer, num_words_title):
    with open(path + "/item_to_name.pkl", "rb") as f:
        item_to_name = pickle.load(f)
    metadata = {}
    with open(path + item_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            item_name, item_title, item_abs = line.split('\t')
            metadata[item_name] = [item_title, item_abs]
    item_id_to_bert_tokenized = {}
    for item_id, item_name in item_to_name.items():
        title = metadata[item_name][0]
        item_id_to_bert_tokenized[item_id] = bert_tokenizer(title.lower(), max_length=num_words_title, padding='max_length', truncation=True)

    max_item_id = max([item_id for item_id in item_to_name.keys()])
    title_input_ids = np.zeros((max_item_id + 1, num_words_title), dtype='int32')
    title_attn_mask = np.zeros((max_item_id + 1, num_words_title), dtype='int32')
    for item_id, title_tokenized in item_id_to_bert_tokenized.items():
        title_input_ids[item_id] = title_tokenized["input_ids"]
        title_attn_mask[item_id] = title_tokenized['attention_mask']
    item_content = np.concatenate([title_input_ids, title_attn_mask], axis=1)

    return item_to_name, item_content

