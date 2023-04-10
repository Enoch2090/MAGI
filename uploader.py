from magi_configs import MagiProductionConfig
from magi_models import *
from indexers import *
from dataset import *
from pprint import pprint
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from terminaltables import AsciiTable
from tqdm.auto import tqdm
from dataclasses import asdict
from magi_dataset import *
import argparse

# Defining helper functions
def upload_to_es(es_instance, data, index:str, batch_size=1000):
    bulk_data = []
    for i, repo in enumerate(tqdm(data)):
        bulk_data.append(
            {
                '_index': index,
                '_id': i,
                '_source': asdict(repo)
            }
        )
        if (i + 1) % batch_size == 0:
            bulk(es_instance, bulk_data)
            bulk_data = []
    bulk(es_instance, bulk_data)
    es_instance.indices.refresh(index=index)
    return es_instance.cat.count(index=index, format='json')   

def print_query(response, additonal_fields=[]):
    table_data = [
        ['name', 'lang', 'link', 'description', 'score', *additonal_fields]
    ]
    for x in response['hits']['hits']:
        table_data.append(
            [x['_source']['name'], x['_source']['lang'], x['_source']['link'], x['_source']['description'][:100], x['_score'], *[x[field] for field in additonal_fields]]
        )
    table = AsciiTable(table_data)
    print(table.table)

if __name__ == '__main__':
    config = MagiProductionConfig()

    parser = argparse.ArgumentParser(description="MAGI Uploader")
    parser.add_argument('--es', action='store_true', help='Whether upload to Elasticsearch')
    parser.add_argument('--pinecone', action='store_true', help='Whether upload to Pinecone')
    args = parser.parse_args()

    if args.pinecone:
        datasets = [
            GitHubCorpusRawTextDataset(c, lang=lang, chunk_size=1024, max_num=4) for c, lang in zip(config.lang_corpus, config.langs)
        ]
        indexer = MagiIndexer(datasets, None, config.embedding_file)
        indexer._upload_pinecone(
            config
        )
        
        model = get_distilbert_base_dotprod('Enoch2090/MAGI').to('cuda')
        q = model.encode(['Python find slowest part of my program'], show_progress_bar=False)[0].tolist()
        index = pinecone.Index('magi-data')
        pprint(
            index.query(
                vector = q,
                top_k = 10,
                namespace = 'Python'
            )
        )
    
    if args.es:
        es = Elasticsearch(
            config.es_url, 
            ca_certs =  './http_ca.crt',
            basic_auth = (config.es_username, config.es_passwd)
        )
        for lang in ['Python', 'C++', 'JavaScript']:
            lang_safe = lang.lower().replace('++', 'pp')
            es.options(ignore_status=400).indices.create(index=f'{lang_safe}-index')
            data = GitHubDataset(empty=False, file_path=f'{lang_safe}-latest')
            print(
                upload_to_es(
                    es, 
                    data, 
                    index = f'{lang_safe}-index', 
                    batch_size = 3000
                )
            )
        resp = es.search(
            index = '*-index', # python-index, cpp-index & javascript-index
            query = {
                'match' : {
                    'readme' : 'web archiving service'
                }
            }
        )
        print_query(resp)