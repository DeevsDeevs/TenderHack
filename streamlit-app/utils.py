import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedTokenizerFast, BertModel, BertConfig
from pyjarowinkler import distance
import faiss
from copy import deepcopy
import time
import json
import nltk 


def clear_text(text, punctuation, morph):
    tokens = nltk.word_tokenize(text, language="ru")
    tokens = [morph.parse(i)[0].normal_form for i in tokens if i not in punctuation]
    return " ".join(tokens)

def get_model_tokenizer(hub_name: str):
    with open("bert_config.json", "r") as read_config:
        config = json.load(read_config)
    bert_config = BertConfig(**config)  
    tokenizer = AutoTokenizer.from_pretrained(hub_name)
    model = BertModel(bert_config)
    return model, tokenizer


class BertCLS(nn.Module):
    def __init__(self, model, n_classes):
        super(BertCLS, self).__init__()
        self.model = model
        self.fc = nn.Linear(768, n_classes)

    def forward(self, batch):
        return self.fc(self.model(**batch).pooler_output)

def get_kpgz(bert_cls, tokenizer, text, kpgz_dict):
    tokens = tokenizer(text, padding=True,
                max_length=200, truncation=True,
                return_tensors='pt')
    with torch.no_grad():
        tokens = tokens.to(bert_cls.model.device)        
        return kpgz_dict[bert_cls(tokens).argmax(-1).detach().cpu().numpy()[0]]

def get_embeddings(bert_cls, tokenizer, text):
    tokens = tokenizer(text, padding=True,
                    max_length=200, truncation=True,
                    return_tensors='pt')

    with torch.no_grad():
        tokens = tokens.to(bert_cls.model.device)
        return bert_cls.model(**tokens).pooler_output.detach().cpu().numpy()

def calc_kpgz(kpgz, kpgz_table):
    s = 0
    for indx, (i, j) in enumerate(zip(kpgz_table.split('.'), kpgz.split('.'))):
        s += 1
        if i != j:
            return indx
    return s + 1

def string_dist(str1, str2):
    return distance.get_jaro_distance(str1, str2,
                                      winkler=True,
                                      winkler_ajustment=True,
                                      scaling=0.2)


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    return data

def filter_for_rec(kpgz, kpgz_table):
    return kpgz != kpgz_table

def get_search_results(search_request: str, additional_info: str, data: pd.DataFrame, bert_cls: BertCLS, index: faiss.IndexFlatIP, tokenizer, kpgz_dict: dict, rec = False, rec_dict = dict(), item_index = None) -> pd.DataFrame:
    new_data = deepcopy(data)
    new_data = prepare_data(data=new_data)

    search_request = search_request
    search_request += " [SEP] " + additional_info
    kpgz_table = get_kpgz(bert_cls, tokenizer, search_request, kpgz_dict)
    if rec:
        new_data['sub_kpgz'] = new_data['Код КПГЗ'].apply(lambda x: filter_for_rec(x, kpgz_table))
    new_data['kpgz_sim'] = new_data['Код КПГЗ'].apply(lambda x: calc_kpgz(x, kpgz_table))
    xq = get_embeddings(bert_cls, tokenizer, search_request)
    faiss.normalize_L2(xq)
    k = 100
    D, I = index.search(xq, k)  # type: ignore

    indexes = set(new_data.index)
    selected = [i for i in I[0] if i in indexes]

    faiss_results = new_data.loc[selected].reset_index(
        drop=True)  # type: ignore
    faiss_results['string_dist'] = faiss_results['Название СТЕ'].apply(
        lambda x: string_dist(x, search_request))

    if rec:
        faiss_results = faiss_results[faiss_results['sub_kpgz'] == True]
        
        if item_index in rec_dict:
            history_rec_ids = rec_dict[item_index]
            history_rec_ids = set(i[0] for i in history_rec_ids)
            history_rec_df = new_data[new_data['ID СТЕ'].isin(history_rec_ids)]
            return pd.concat([history_rec_df, faiss_results.sort_values(by=['kpgz_sim', 'string_dist'], ascending=[False, False]).head(5)])


        return faiss_results.sort_values(by=['kpgz_sim', 'string_dist'], ascending=[False, False]).head(5)

    return faiss_results.sort_values(by=['kpgz_sim', 'string_dist'], ascending=[False, False]).head(5)