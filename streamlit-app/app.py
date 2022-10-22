from typing import final
import pandas as pd
import numpy as np
from yaml import load
import utils as utils
import torch
import faiss
import onnxruntime as rt
import matplotlib.pyplot as plt
import json

import streamlit as st
import st_aggrid as st_agg

import pymorphy2
import nltk
import string


@st.cache(allow_output_mutation=True)
def load_all():
    model, tokenizer = utils.get_model_tokenizer("cointegrated/LaBSE-en-ru")
    model.eval()
    device = torch.device('cpu')

    bert_cls = utils.BertCLS(model, 5307)
    bert_cls.load_state_dict(torch.load(
        "BertCLS_epoch_5.pth", map_location=device))
    bert_cls = bert_cls.to(device)

    final_df = pd.read_feather(
        "preprocess_all_columns_lem.feather")
    price_df = pd.read_feather(
        "median_prices.feather").rename(columns={'median': 'price'})
    data = final_df[final_df.columns[:-768]].reset_index(drop=True)
    data = pd.merge(data, price_df.rename(columns={'id': 'ID СТЕ'}), on='ID СТЕ', how='left')

    rec_dict = dict(pd.read_parquet('rec_data.parquet').values)

    embeddings = final_df[final_df.columns[-768:]].values.astype(np.float32)
    embeddings = np.ascontiguousarray(embeddings)

    faiss.normalize_L2(embeddings)
    d = 768 
    index = faiss.IndexFlatIP(d)
    index.add(embeddings) # type: ignore

    kpgz_dict = dict(final_df[['target', 'Код КПГЗ']].values)

    punctuation = set(string.punctuation)
    morph = pymorphy2.MorphAnalyzer()

    return data, tokenizer, bert_cls, index, kpgz_dict, rec_dict, punctuation, morph

# def get_fig_price(series: pd.Series):
#     try: 
#         data = {"11-20": series[(series.quantile(0.11) <= series) & (series <= series.quantile(0.2))].mean(),
#                 "21-30": series[(series.quantile(0.21) <= series) & (series <= series.quantile(0.3))].mean(),
#                 "31-40": series[(series.quantile(0.31) <= series) & (series <= series.quantile(0.4))].mean(),
#                 "41-50": series[(series.quantile(0.41) <= series) & (series <= series.quantile(0.5))].mean(),
#                 "51-60": series[(series.quantile(0.51) <= series) & (series <= series.quantile(0.6))].mean(),
#                 "61-70": series[(series.quantile(0.61) <= series) & (series <= series.quantile(0.7))].mean(),
#                 "71-80": series[(series.quantile(0.71) <= series) & (series <= series.quantile(0.8))].mean(),
#                 "81-90": series[(series.quantile(0.81) <= series) & (series <= series.quantile(0.9))].mean(), }
#         courses = list(data.keys())
#         values = list(data.values())
#         plt.style.use('dark_background')  # type: ignore
#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.bar(courses, values, width=0.4)
#         ax.bar_label(ax.containers[0])  # type: ignore
#         plt.xlabel("Quantile Group")
#         plt.ylabel("Price")
#         plt.title("Распределение средней цены по квантилям")
#         plt.axhline(y=(data['11-20'] + data['81-90']) / 2,
#                     linewidth=3, color='red', label="Средняя цена", ls="--")
#         plt.axhline(y=series.median(), linewidth=3, color='pink',
#                     label="Медианная цена", ls="--")
#         plt.legend()
#         return fig
#     except:
#         return None

# def get_fig_tops(series: pd.Series):
#     try:
#         plt.style.use('dark_background')  # type: ignore
#         fig, ax = plt.subplots(figsize=(7, 3))
#         counts = series.value_counts()[:5]
#         counts = counts / len(series) * 100
#         counts.plot(ax = ax, kind = 'barh', xlabel = 'Процент предложений выставленных этим ИНН от общего числа')
#         ax.bar_label(ax.containers[0])  # type: ignore
#         return fig
#     except:
#         return None


def main():
    st.set_page_config(page_title='Tender Search Engine')
    st.markdown("""
    # Tender Search Engine 
    """)
    data, tokenizer, bert_cls, index, kpgz_dict, rec_dict, punctuation, morph = load_all()
    search_request = st.text_input('Введите слова для поиска:').lower().strip()
    search_expander = st.expander('Дополнительные настройки')
    additional_info = search_expander.text_input('Дополнительные параметры').lower().strip()
    if search_request:
        search_request = utils.clear_text(search_request, punctuation, morph)
        search_results = utils.get_search_results(search_request=search_request, additional_info=additional_info, data=data, 
                                                  bert_cls = bert_cls, index=index, tokenizer=tokenizer, kpgz_dict = kpgz_dict)
        gb_main = st_agg.GridOptionsBuilder.from_dataframe(search_results)
        gb_main.configure_default_column(
            groupable=True, value=True, enableRowGroup=True, editable=False)
        gb_main.configure_side_bar()
        gb_main.configure_selection('single', use_checkbox=True, )
        gb_main.configure_pagination(
            paginationPageSize=10, paginationAutoPageSize=False)
        gb_main.configure_grid_options(domLayout='normal')
        grid_main_options = gb_main.build()
        grid_main_response = st_agg.AgGrid(
            search_results,
            gridOptions=grid_main_options,
            width='100%',
            update_mode=st_agg.GridUpdateMode.MODEL_CHANGED,
            data_return_mode=st_agg.DataReturnMode.AS_INPUT,
            key='main',
            reload_data=True,
        )
        selected_main_row = grid_main_response['selected_rows']
        if len(selected_main_row) != 0:
            recommend_results = utils.get_search_results(search_request=selected_main_row[0]['Название СТЕ'].strip().lower(), additional_info=additional_info, data=data, 
                                                  bert_cls = bert_cls, index=index, tokenizer=tokenizer, kpgz_dict = kpgz_dict, rec = True, rec_dict=rec_dict, item_index=selected_main_row[0]['ID СТЕ'])
            gb_rec = st_agg.GridOptionsBuilder.from_dataframe(search_results)
            gb_rec.configure_default_column(
                groupable=True, value=True, enableRowGroup=True, editable=False)
            gb_rec.configure_side_bar()
            gb_rec.configure_selection('single', use_checkbox=False, )
            gb_rec.configure_pagination(
                paginationPageSize=10, paginationAutoPageSize=False)
            gb_rec.configure_grid_options(domLayout='normal')
            grid_rec_options = gb_rec.build()
            st.markdown("""
            ### Сопутствующие товары
            """)
            grid_rec_response = st_agg.AgGrid(
                recommend_results,
                gridOptions=grid_rec_options,
                width='100%',
                update_mode=st_agg.GridUpdateMode.MODEL_CHANGED,
                data_return_mode=st_agg.DataReturnMode.AS_INPUT,
                key='rec',
                reload_data=True,
            )   


if __name__ == '__main__':
    main()
