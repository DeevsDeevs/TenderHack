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
import scipy
from transliterate import translit
import string
import pymorphy2
import re
from rank_bm25 import BM25Okapi
from typing import List



def clear_text(text, punctuation, morph):
    tokens = nltk.word_tokenize(text, language="ru")
    tokens = [morph.parse(i)[0].normal_form for i in tokens if i not in punctuation]
    return " ".join(tokens)

def has_only_latin_letters(name):
    char_set = string.ascii_letters
    return all((True if x in char_set else False for x in name))

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

class BM25Model:
    def __init__(self, max_k=100, src = None):
        self.morph = pymorphy2.MorphAnalyzer()
        self.bm25 = None
        self.max_k = max_k
        if src:
            self.tokenized_corpus = np.load(src, allow_pickle=True).tolist()
        else:
            self.tokenized_corpus = None
 
    def clear_text(self, text: str):
        text = re.sub(r'[",*+!:;<=>?@]^_`{|}~]', '', text)
        text = re.sub(r'\s\d+\s', '', text)
        return text
        
    def lemmatize(self, text):
        words = text.split() # разбиваем текст на слова
        res = list()
        for word in words:
            p = self.morph.parse(word)[0]
            nf = p.normal_form  # type: ignore
            if nf not in res:
                res.append(nf)

        return list(res)
    
    def fit(self, corpus: List[str] = []):
        if self.tokenized_corpus is None:
            self.tokenized_corpus = [self.lemmatize(self.clear_text(sent)) for sent in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
    def predict(self, query: str, top_k=None):
        '''Return top_k relevant indexes and their scores'''
        
        if self.bm25 is None:
            raise Exception("Fit model at first!")
            
        tokenized_query = self.lemmatize(self.clear_text(query))
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        if top_k is None:
            n_positive = np.sum(np.array(doc_scores) > 0)
            top_k = np.max([n_positive, self.max_k])  # type: ignore
            
        ind = np.argsort(doc_scores)[top_k:][::-1]
        scores = sorted(doc_scores, reverse=True)[:top_k]
        return ind, scores


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



def prepare_data(data: pd.DataFrame, min_price=0.0, max_price=float('inf'), kpgz_code = "") -> pd.DataFrame:
    if max_price != 0.0:
        data = data.loc[(data['price'] >= min_price) &
                        (data['price'] <= max_price)]
    if  kpgz_code != "":
        data = data.loc[data['Код КПГЗ'].str.startswith(kpgz_code)]
    return data

def filter_for_rec(kpgz, kpgz_table):
    return kpgz != kpgz_table

def get_search_results(search_request: str, additional_info: str, data: pd.DataFrame, bert_cls: BertCLS, embeddings, index: faiss.IndexFlatIP, tokenizer, kpgz_dict: dict, rec = False, rec_dict = dict(), item_id = None, min_price=0.0, max_price=float('inf'), kpgz_code="", trans = False, bm25 = None, goods_df = pd.DataFrame()) -> pd.DataFrame:
    new_data = deepcopy(data)
    new_data = prepare_data(data=new_data, min_price=min_price, max_price=max_price, kpgz_code=kpgz_code)
    kpgz_table = get_kpgz(bert_cls, tokenizer, search_request, kpgz_dict)
    if rec:
        new_data['sub_kpgz'] = new_data['Код КПГЗ'].apply(lambda x: filter_for_rec(x, kpgz_table))
    new_data['kpgz_sim'] = new_data['Код КПГЗ'].apply(lambda x: calc_kpgz(x, kpgz_table))
    xq = get_embeddings(bert_cls, tokenizer, search_request)
    bm25_top10 = pd.DataFrame()
    if bm25 != None:
        ids, scores = bm25.predict(search_request)
        bm25_top10 = goods_df.iloc[ids].head(10)['ID СТЕ'].values # type: ignore

    faiss.normalize_L2(xq)
    k = 100
    D, I = index.search(xq, k)  # type: ignore

    indexes = set(new_data.index)
    selected = [i for i in I[0] if i in indexes]

    faiss_results = new_data.loc[selected].reset_index(
        drop=True)  # type: ignore
        

    faiss_results['cos_sim'] = list(torch.nn.functional.cosine_similarity(torch.from_numpy(embeddings[selected]), torch.from_numpy(xq)).cpu().detach().numpy())
    if not rec:
        bm_data = new_data[((new_data['ID СТЕ'].isin(bm25_top10)))].reset_index(drop=True)
        bm_data['cos_sim'] = 1
        faiss_results = pd.concat([faiss_results, bm_data])

    faiss_results['additional_dist'] = faiss_results['Характеристики'].apply(lambda x: sum((tok in x) for tok in  nltk.word_tokenize(additional_info, language="ru")))
    faiss_results['string_dist'] = faiss_results['Название СТЕ'].apply(
        lambda x: string_dist(x, search_request))

    a = faiss_results['cos_sim'].to_list()
    conf_int = scipy.stats.norm.interval(0.95, loc=np.mean(a), scale=scipy.stats.sem(a))[0]
    conf_int = max(conf_int, 0.65)
    faiss_results = faiss_results.loc[faiss_results['cos_sim'] >= conf_int]

    if rec:
        faiss_results = faiss_results[faiss_results['sub_kpgz'] == True]
        
        if item_id in rec_dict:
            history_rec_ids = rec_dict[item_id]
            history_rec_ids = sorted(history_rec_ids, key=lambda x: x[1], reverse=True)[:10]
            history_rec_ids = set(i[0] for i in history_rec_ids)
            history_rec_df = new_data[new_data['ID СТЕ'].isin(history_rec_ids)]
            return pd.concat([history_rec_df, faiss_results.sort_values(by=['kpgz_sim', 'string_dist'], ascending=[False, False]).head(10)])

        return faiss_results.sort_values(by=['kpgz_sim', 'string_dist'], ascending=[False, False]).head(10)
    
    if len(faiss_results) == 0 and not trans:
        ru_search_request = translit(search_request, 'ru')
        return get_search_results(search_request=ru_search_request, additional_info=additional_info, data=data, 
                                                  bert_cls = bert_cls, embeddings=embeddings, index=index, tokenizer=tokenizer, kpgz_dict = kpgz_dict,
                                                  min_price=min_price, max_price=max_price, kpgz_code=kpgz_code, trans=True)

    return faiss_results.sort_values(by=['additional_dist', 'kpgz_sim', 'string_dist'], ascending=[False, False, False]).head(10)