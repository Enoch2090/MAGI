import streamlit as st
import os
import sys
import time
import json
import random
from magi_models import *
from dataset import *
from indexers import *
GH_TOKEN = ''

# ----------------Functionalities----------------
def render_html(html):
    st.markdown(f'{html}', unsafe_allow_html=True)

@st.experimental_singleton
class CachedDataset(GitHubCorpusRawTextDataset): pass

@st.experimental_singleton
class CachedIndexer: 
    def __init__(self, _dataset, _model):
        self.indexer = MagiIndexer(_dataset, _model, embedding_file='./datafile/msmarco-distilbert-base-dot-prod-v3_trained_embeddings.npy')
    def search(self, *args, **kwargs):
        return self.indexer.search(*args, **kwargs)
    
@st.experimental_memo
def get_model():
    return get_distilbert_base_dotprod('./datafile/ghv5-model')

@st.experimental_memo
def get_sample_queries():
    with open('./datafile/queries.txt', 'r') as f:
        samples = [x[0].capitalize() for x in json.load(f)]
    return samples

def display_results(results):
    for result in results:
        st.markdown(f"🗂  [{result[0]}]({result[1]})")
        st.markdown(f"⭐️  {result[2]} | {result[3]}")
# ----------------Options----------------
def option_query(sample):
    st.title("Search for a package")
    query = st.text_input('Enter query', placeholder=sample)
    st.markdown('Notice: current version of MAGI only supports Python packages. More languages on the way!')
    if st.button("Search"):
        with st.spinner("Querying..."):
            results, retrieve_time = indexer.search(query, rank=10)
            display_results(results)
            st.markdown(f'Retrieved in {retrieve_time:.4f} seconds with {device} backend')
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
dataset = CachedDataset('./datafile/ghv6.json', mode='index', chunk_size=1024, max_num=4)
model = get_model()
indexer = CachedIndexer(dataset, model)
samples = get_sample_queries()

if option == 'Query':
    option_query(random.sample(samples, 1)[0])
elif option == 'About':
    option_about()


# ----------------Hide Development Menu----------------
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
