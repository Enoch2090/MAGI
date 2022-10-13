import streamlit as st
import os
import sys
import time
import json
import gc
import random
import logging
import requests
from tqdm import tqdm, trange
# from magi_models import *
from dataset import *
from indexers import *
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

GH_TOKEN = ''
LANGS = ['Python', 'JavaScript', 'C++']

st.set_page_config(
   page_title='MAGI',
   page_icon='🗃',
   initial_sidebar_state='expanded',
)

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

@st.experimental_singleton
class CachedDataset(GitHubCorpusRawTextDataset): pass

@st.experimental_singleton
class CachedIndexer: 
    def __init__(self, _dataset, _model):
        self.indexer = MagiIndexer(_dataset, _model, embedding_file='./datafile/msmarco-distilbert-base-dot-prod-v3_ghv7.pkl')
    def search(self, *args, **kwargs):
        return self.indexer.search(*args, **kwargs)
    
@st.experimental_memo
def get_model():
    # return get_distilbert_base_dotprod('Enoch2090/MAGI')
    model = ProductionModel(os.getenv('HUGGINGFACE_TOKEN'))
    print(model.headers)
    return model

@st.experimental_memo
def get_sample_queries():
    samples = []
    with open('./datafile/queries.txt', 'r') as f:
        queries = json.load(f)
        for lang in queries.keys():
            samples += [(x[0].capitalize(), lang) for x in queries[lang]]
    return samples

def display_results(results):
    st.markdown('''<hr style="height:2px;border:none;color:#CCC;background-color:#CCC;" />''', unsafe_allow_html=True)
    for result in results:
        st.markdown(f"🗂  [{result[0]}]({result[1]})")
        st.markdown(f"⭐️  {result[2]} | {result[3]}")
        st.markdown('''<hr style="height:2px;border:none;color:#CCC;background-color:#CCC;" />''', unsafe_allow_html=True)

def run_query(query, lang):
    with st.spinner("Querying..."):
        try:
            st.markdown(f'Results for "{query}" in `{lang}`')
            results, retrieve_time = indexer.search(query, lang=lang, rank=10)
            display_results(results)
            st.markdown(f'Retrieved in {retrieve_time:.4f} seconds with {device} backend')
        except CloudLoadingException:
            st.markdown(f'Cloud model is currently loading, please retry after 30 seconds.')
        
# ----------------Options----------------
def option_query(samples):
    st.title("Search for a package")
    query = st.text_input('Enter query', help='Describe what functionality you are looking for', max_chars=2048)
    lang = st.selectbox(
        'Search in language...',
        tuple(LANGS)
    )
    col1, col2 = st.columns(2)
    with col1:
        search = st.button("Search")
    with col2:
        lucky = st.button("Feeling lucky")
    if search:
        run_query(query, lang)
    elif lucky:
        sample = random.sample(samples, 1)[0]
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
if not os.path.exists('ghv7_transformed.json'):
    get_corpus('https://huggingface.co/datasets/Enoch2090/github_semantic_search/resolve/main/ghv7_transformed.json')
datasets = [
    CachedDataset('./ghv7_transformed.json', lang=lang, chunk_size=1024, max_num=4) for lang in LANGS
]
model = get_model()
indexer = CachedIndexer(datasets, model)
samples = get_sample_queries()

if option == 'Query':
    option_query(samples)
elif option == 'About':
    option_about()


# ----------------Hide Development Menu----------------
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
