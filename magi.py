from turtle import onclick
import streamlit as st
import os
import sys
import time
import json
import gc
import random
import logging
import requests
import urllib3
import uuid
from tqdm import tqdm, trange
from itertools import zip_longest, chain
from magi_models import *
from dataset import *
from indexers import *
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from enum import Enum

from magi_configs import MagiProductionConfig


logging.basicConfig()
logger = logging.getLogger('MAGI_interface')
logger.setLevel(logging.INFO)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(
   page_title='MAGI',
   page_icon='🗃',
   initial_sidebar_state='expanded',
)

config = MagiProductionConfig()


# ----------------Functionalities----------------
def render_html(html):
    st.markdown(f'{html}', unsafe_allow_html=True)
    
def get_corpus(link):
    local_file_name = link.split('/')[-1]    
    r = requests.get(link, stream=True)
    file_size = int(r.headers.get('content-length'))
    logger.info(f'Downloading {link} as {local_file_name}')
    logger.info(f'Detected file size {file_size} kb.')
    with open(local_file_name, "wb") as f:
        with tqdm(total = file_size // 1024) as _tqdm:
            chunk_n = 0
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                chunk_n += 1
                _tqdm.update(1)

def interleave_rerank(src1: List[Tuple], src2: List[Tuple]) -> List[Tuple]:
    # TODO: remove duplicates
    results = []
    repo_hist = []
    for s in chain(*zip_longest(src1, src2, fillvalue=None)):
        if s is None or s[0] in repo_hist:
            continue
        results.append(s)
        repo_hist.append(s[0])
    return results

def get_ranking_dict(ranking_results):
    return {
          'id': str(uuid.uuid4()),
          'timestamp': datetime.timestamp(datetime.now()),
          'user': st.session_state['usr_id'],
          'session': st.session_state['usr_id'],
          'fields': [],
          'items': [
            {
              'id': str(x[6]),
              x[5]: float(x[4])
            } for x in ranking_results
          ],
          'event': 'ranking'
        } 

def get_interaction_dict(item_id, ranking_id):
    return {
        'id': str(uuid.uuid4()),
        'item': str(item_id),
        'timestamp': datetime.timestamp(datetime.now()),
        'ranking': ranking_id,
        'user': st.session_state['usr_id'],
        'session': st.session_state['usr_id'],
        'type': 'click',
        'fields': [],
        'event': 'interaction'
    }

def callback_save_interaction(item_id):
    interaction_dict = get_interaction_dict(item_id, st.session_state['ranking_id'])
    with open('magi_interactions.jsonl', 'a') as f:
        f.write(json.dumps(interaction_dict))
        f.write('\n')

@st.experimental_singleton
class CachedDataset(GitHubCorpusRawTextDataset): pass

@st.experimental_singleton
class CachedElasticsearch(Elasticsearch): pass

@st.experimental_singleton
class CachedIndexer: 
    def __init__(self, _dataset, _model):
        self.indexer = MagiIndexer(_dataset, _model, embedding_file=config.embedding_file)
    def search(self, *args, **kwargs):
        return self.indexer.search(*args, **kwargs)

@st.experimental_memo
def get_model():
    if config.device in ['cuda', 'cpu']:
        return get_distilbert_base_dotprod('Enoch2090/MAGI').to(config.device)
    elif config.device == 'hf':
        model = ProductionModel(os.getenv('HUGGINGFACE_TOKEN'))
        print(model.headers)
    return model

@st.experimental_memo
def get_sample_queries(lang=None):
    samples = []
    with open('./datafile/queries.txt', 'r') as f:
        queries = json.load(f)
        if lang:
            return [(x[0].capitalize(), lang) for x in queries[lang]]
        for lang in queries.keys():
            samples += [(x[0].capitalize(), lang) for x in queries[lang]]
    return samples

def display_results(results):
    st.session_state['app_state'] = 'WAIT'
    if st.button('Return'):
        st.session_state['app_state'] = 'STANDBY'
        st.session_state['latest_results'] = None
        st.experimental_rerun()
    st.markdown(f'Results for "{st.session_state["latest_query"]["query"]}" in `{st.session_state["latest_query"]["lang"]}`')
    st.markdown('''<hr style="height:1.5px;border:none;color:#CCC;background-color:#CCC;" />''', unsafe_allow_html=True)
    for index, result in enumerate(results):
        col1, col2 = st.columns([1, 20], gap = 'medium')
        with col1:
            st.button(label = '▲', key = f'upvote-{index}', on_click = callback_save_interaction, args = [result[6]])
        with col2:
            st.markdown(f'[**{result[0]}**]({result[1]})')
        st.markdown(f'⭐️  {result[2]} | {result[3]}')
        st.markdown('''<hr style="height:1.5px;border:none;color:#CCC;background-color:#CCC;" />''', unsafe_allow_html=True)

def run_query(query, lang):
    lang_safe = lang.lower().replace('++', 'pp')
    try:
        with st.spinner("Querying..."):
            sim_results, retrieve_time = indexer.search(query, lang=lang, rank=10)
            es_resp = es.search(
                index = f'{lang_safe}-index',
                query = {
                    'match' : {
                        'readme': query
                    }
                }
            )
            es_results = [(x['_source']['name'], x['_source']['link'], x['_source']['stars'], x['_source']['description'], float(x['_score']), 'bm25', x['_id']) for x in es_resp.body['hits']['hits']]
            results = interleave_rerank(sim_results, es_results)
            ranking_dict = get_ranking_dict(results)
            with open('magi_interactions.jsonl', 'a') as f:
                f.write(json.dumps(ranking_dict))
                f.write('\n')
            st.session_state['ranking_id'] = ranking_dict['id']
            st.session_state['latest_query'] = {
                'lang': lang,
                'query': query,
                'results': results
            }
            display_results(results)
            st.markdown(f'Retrieved in {retrieve_time:.4f} seconds with {device} backend')
    except CloudLoadingException:
        st.markdown(f'Cloud model is currently loading, please retry after 30 seconds.')
        my_bar = st.progress(0)
        for percent_complete in range(33):
            time.sleep(1)
            my_bar.progress(percent_complete + 3)
        
# ----------------Options----------------
def option_query(sample_dict):
    if (st.session_state['app_state'] == 'WAIT'):
        display_results(st.session_state['latest_query']['results'])
        gc.collect()
        return
    st.title("Search for a package")
    query = st.text_input('Enter query', help='Describe what functionality you are looking for', max_chars=2048)
    lang = st.selectbox(
        'Search in language...',
        tuple(config.langs)
    )
    col1, col2 = st.columns(2)
    with col1:
        search = st.button("Search")
    with col2:
        lucky = st.button("Feeling lucky")
    if search:
        run_query(query, lang)
    elif lucky:
        sample = random.sample(sample_dict[lang], 1)[0]
        run_query(*sample)
    gc.collect()
    return


def option_about():
    with open('README.md', 'r') as f:
        readme = "".join(f.readlines())
    st.markdown(readme, unsafe_allow_html=True)

# ----------------Menu----------------
st.sidebar.title('MAGI: An semantic searcher over GitHub')
option = st.sidebar.selectbox(
            'Menu',
            ['Query', 'About']
        )
datasets = [
    CachedDataset(c, lang=lang, chunk_size=1024, max_num=4) for c, lang in zip(config.lang_corpus, config.langs)
]
model = get_model()
indexer = CachedIndexer(datasets, model)
es = CachedElasticsearch(
    config.es_url, 
    ca_certs =  config.es_cert,
    basic_auth = (config.es_username, config.es_passwd),
    verify_certs=False,
)
if 'app_state' not in st.session_state.keys():
    st.session_state['app_state'] = 'STANDBY'
if 'usr_id' not in st.session_state.keys():
    st.session_state['usr_id'] = str(uuid.uuid4())
    
# samples = get_sample_queries()
sample_dict = {lang: get_sample_queries(lang) for lang in config.langs}

if option == 'Query':   
    option_query(sample_dict)
elif option == 'About':
    option_about()


# ----------------Hide Development Menu----------------
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
