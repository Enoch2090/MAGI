from dataclasses import dataclass
import os
@dataclass
class MagiProductionConfig:
    gh_token = os.getenv('GH_TOKEN') 
    es_url = 'https://52.87.231.111:9200'
    es_username = 'elastic'
    es_passwd = os.getenv('ES_PASSWD')
    es_cert = './http_ca.crt'
    langs = ['Python', 'JavaScript', 'C++', 'Rust', 'Go']
    # lang_corpus = ['python-latest', 'javascript-latest', 'cpp-latest', 'rust-latest', 'go-latest']
    lang_corpus = ['./magi_downloads/ghv10_python.json', './magi_downloads/ghv10_javascript.json', './magi_downloads/ghv10_cpp.json', './magi_downloads/ghv10_rust.json', './magi_downloads/ghv10_go.json']
    embedding_file = './datafile/MAGI_ghv10.pkl'
    device = 'cuda'
    