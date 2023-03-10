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

def es_search(es, query: str, lang_safe: str):
    es_resp = es.search(
        index = f'{lang_safe}-index',
        query = {
            'match' : {
                'readme': query
            }
        }
    )
    return [
        {
            'index': x['_id'],
            'name': x['_source']['name'],
            'link': x['_source']['link'],
            'stars': x['_source']['stars'],
            'description': x['_source']['description'],
            'metric': 'bm25',
            'value': float(x['_score'])
        } for x in es_resp.body['hits']['hits']
    ]

def interleave_rerank(src1: List[dict], src2: List[dict], query:str = '', lang: str = '') -> List[dict]:
    results = []
    repo_hist = []
    for s in chain(*zip_longest(src1, src2, fillvalue=None)):
        if s is None:
            continue
        if s['name'] in repo_hist:
            results[repo_hist.index(s['name'])][s['metric']] = s['value']
            continue
        s[s['metric']] = s['value']
        results.append(s)
        repo_hist.append(s['name'])
    return results

def metarank_rerank(src1: List[dict], src2: List[dict], query:str = '', lang: str = '') -> List[dict]:
    merged_results = interleave_rerank(src1, src2)
    inverse_dict = {x['index']: x for x in merged_results}
    ranking_dict = get_ranking_dict(merged_results, query, lang)
    metarank_raw_result = requests.post(f'{config.metarank_url}/rank/{config.metarank_model}', json = ranking_dict).json()['items']
    # requests.post(f'{config.metarank_url}/feedback', json = ranking_dict)
    results = []
    for ranked_item in metarank_raw_result:
        repo_item = inverse_dict[ranked_item['item']]
        repo_item['ranked_score'] = ranked_item['score']
        results.append(repo_item)
    return results
    

def get_ranking_dict(ranking_results, query:str = '', lang: str = ''):
    return {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'user': st.session_state['usr_id'],
            'session': st.session_state['usr_id'],
            'fields': [
                {'name': 'query', 'value': query},
                {'name': 'lang', 'value': lang}
            ],
            'items': [
                {
                    'id': str(x['index']),
                    'fields': [
                        {
                            'name': 'stars',
                            'value': x['stars']
                        },
                        *[
                            {
                                'name': metric,
                                'value': x[metric]
                            } for metric in ['bm25', 'similarity'] if metric in x.keys()
                        ] 
                    ]
                } for x in ranking_results
            ],
            'event': 'ranking'
        } 

def get_interaction_dict(item_id, ranking_id):
    return {
        'id': str(uuid.uuid4()),
        'item': str(item_id),
        'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'ranking': ranking_id,
        'user': st.session_state['usr_id'],
        'session': st.session_state['usr_id'],
        'type': 'click',
        'fields': [],
        'event': 'interaction'
    }

def callback_save_interaction(item_id, feedback = False):
    interaction_dict = get_interaction_dict(item_id, st.session_state['ranking_id'])
    if item_id in st.session_state['upvoted']:
        return
    st.session_state['upvoted'].append(item_id)
    if feedback:
        requests.post(f'{config.metarank_url}/feedback', json = interaction_dict)
        return
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
        upvote_text = '' if not result['index'] in st.session_state['upvoted'] else '<font color="#f5675f">    Upvoted!</font>'
        with col1:
            st.button(label = '▲', key = f'upvote-{index}', on_click = callback_save_interaction, args = [result['index']])
        with col2:
            st.markdown(f"[**{result['name']}**]({result['link']}){upvote_text}", unsafe_allow_html=True)
        st.markdown(f"⭐️  {result['stars']} | {result['description']}")
        st.markdown('''<hr style="height:1.5px;border:none;color:#CCC;background-color:#CCC;" />''', unsafe_allow_html=True)

def run_query(query, lang):
    lang_safe = lang.lower().replace('++', 'pp')
    try:
        with st.spinner("Running dense retrieval..."):
            sim_results, retrieve_time = indexer.search(query, lang=lang, rank=10)
        with st.spinner("Running sparse retrieval..."):
            es_results = es_search(es, query, lang_safe)
        with st.spinner("Generating final results..."):
            # results = interleave_rerank(sim_results, es_results)
            results = metarank_rerank(sim_results, es_results, query, lang)
            ranking_dict = get_ranking_dict(results, query, lang)
            print(ranking_dict)
            with open('magi_interactions.jsonl', 'a') as f:
                f.write(json.dumps(ranking_dict))
                f.write('\n')
        st.session_state['ranking_id'] = ranking_dict['id']
        st.session_state['latest_query'] = {
            'lang': lang,
            'query': query,
            'results': results
        }
        st.session_state['upvoted'] = []
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
        label = 'Search in language...',
        options = tuple(config.langs),
        index = st.session_state['last_lang']
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
    st.session_state['last_lang'] = config.langs.index(lang)
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
if 'last_lang' not in st.session_state.keys():
    st.session_state['last_lang'] = 0
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
        
floating_button = """
<style>

#myBtn {
  display: none;
  position: fixed;
  bottom: 20px;
  right: 30px;
  z-index: 99;
  font-size: 18px;
  border: none;
  outline: none;
  background-color: red;
  color: white;
  cursor: pointer;
  padding: 15px;
  border-radius: 4px;
}

#myBtn:hover {
  background-color: #555;
}
</style>

<body>

<button onclick="topFunction()" id="myBtn" title="Go to top">Top</button>

<script>
// Get the button
let mybutton = document.getElementById("myBtn");

// When the user scrolls down 20px from the top of the document, show the button
window.onscroll = function() {scrollFunction()};

function scrollFunction() {
  if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
    mybutton.style.display = "block";
  } else {
    mybutton.style.display = "none";
  }
}

// When the user clicks on the button, scroll to the top of the document
function topFunction() {
  document.body.scrollTop = 0;
  document.documentElement.scrollTop = 0;
}
</script>

</body>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)
st.markdown(floating_button, unsafe_allow_html=True)
