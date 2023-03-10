from dataset import GitHubCorpusRawTextDataset, get_hash
from typing import List, Union, Tuple, Dict
from github import Github
from tqdm.auto import tqdm
import numpy as np
import time
import json
import os
import logging
import pickle
import hashlib
import requests

logger = logging.getLogger('MAGI_training')

class RepoName(str): pass
class RepoDescription(str): pass
class RepoLink(str): pass
class RepoStars(int): pass
class SearchScore(int): pass
class CloudLoadingException(Exception): pass
GithubQueryResult = Tuple[RepoName, RepoLink, RepoStars, RepoDescription]
TestCase = Tuple[str, List[RepoName]]

try:
    GH_TOKEN = os.getenv('GH_TOKEN') 
    # see https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token
except:
    GH_TOKEN = ''    
    
class ProductionModel:
    # uses model hosted on Huggingface API for inference. This class is abstracted to be similar to a Transformer model.
    API_URL = "https://api-inference.huggingface.co/models/Enoch2090/MAGI"
    def __init__(self, token):
        self.headers = {
            "Authorization": token
        }
    def query(self, payload):
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        try: 
            return response.json()['error']
        except:
            return response.json()
    def encode(self, query, show_progress_bar=False):
        data = self.query({
            "inputs": query,
        })
        if type(data) is str:
            raise(CloudLoadingException('Cloud model not ready'))
        else:
            return np.array(data)
    def similarity_func(self, query_embedding, pooled_embeddings):
        return (pooled_embeddings @ query_embedding.T).squeeze(axis=1)

class MagiIndexer:
    def __init__(
        self, 
        datasets: List[GitHubCorpusRawTextDataset], 
        model, 
        embedding_file: str = None
    ) -> Tuple[List[GithubQueryResult], int]:
        '''
        Arguments:
        - datasets: A list of GitHubCorpusRawTextDataset, each representing repositories of a specific programming language.
        - model: A SentenceTransformer model with its similarity_func method overrided with a similarity function (e.g. model.similarity_func = magi_models.tensor_dot_prod_similiarity), or a indexers.ProductionModel.
        - embedding_file: Embeddings generated with indexers.cache_embeddings. Must match the version of datasets, otherwise the result is not guranteed.
        '''
        self.datasets = {d.lang: d for d in datasets}
        self.langs = [d.lang for d in datasets]
        self.model = model
        self.embeddings = {}
        if embedding_file is not None:
            # FIXME:
            # Checksum validation temporaily disabled.
            # Implement checksum validation in magi_dataset in the future
            # checksum = get_hash(self.datasets[self.langs[0]].file_dir)
            with open(embedding_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            # assert self.embeddings['MD5_checksum'] == checksum, 'Embedding is not generated from the same dataset'
            # logger.info(f'Loaded embeddings from {embedding_file}, checksum={checksum} passed')
            logger.info(f'Loaded embeddings from {embedding_file}')
            lang = self.langs[-1]
        else:
            # self.embeddings['MD5_checksum'] = get_hash(self.datasets[self.langs[0]].file_dir)
            for lang in self.langs:
                self.embeddings[lang] = self.model.encode([x[0] for x in self.datasets[lang]])
            # logger.info(f'Generated new embeddings, checksum={self.embeddings["MD5_checksum"]} saved')
            logger.info(f'Generated new embeddings saved')
        # All embedding matrices share the same width, so just take the last one for d
        _, d = self.embeddings[lang].shape
        self.pooled_embeddings = {
            lang: np.zeros(
                (self.datasets[lang].repo_num, d)
            ) for lang in self.langs
        }
        for lang in self.langs:
            for index, (repo, repo_index) in enumerate(self.datasets[lang].repo_to_vec.items()):
                repo_pooling = self.embeddings[lang][repo_index].mean(axis=0)
                self.pooled_embeddings[lang][index, :] = repo_pooling
            l, d = self.pooled_embeddings[lang].shape
            print(f'Language {lang}, pooled total {l} vectors, d = {d}')
            self.pooled_embeddings[lang] = self.pooled_embeddings[lang].astype(np.float32)
    
    def get_repo(
        self, 
        index: int, 
        lang: str
    ):
        return self.datasets[lang].get_repo(repo_index=index)

    def search(
        self, 
        query: str, 
        lang: str, 
        rank=10
    ):
        start = time.time()
        query_embedding = self.model.encode([query], show_progress_bar=False)
        similarity = self.model.similarity_func(query_embedding, self.pooled_embeddings[lang])
        unsorted_index = np.argpartition(similarity, -rank)[-rank:]
        # unsorted_index = np.argsort(similarity)
        sorted_index = unsorted_index[np.flip(np.argsort(similarity[unsorted_index]))]
        results = []
        for index in sorted_index:
            results.append(
                {
                    'index': str(index),
                    **self.get_repo(index, lang),
                    'metric': 'similarity',
                    'value': float(similarity[index])
                }
            )
        end = time.time()
        runtime = float(end - start)
        return results, runtime    
    
class GitHubSearcher:
    # Comparison searcher using GitHub's API to perform plain text search
    def __init__(
        self, 
        token: str
    ):
        self.github_client = Github(token)
        
    def search(
        self, 
        query, 
        lang, 
        rank=10
    ) -> Tuple[List[GithubQueryResult], int]:
        repositories = self.github_client.search_repositories(query=f'{query} stars:>10 language:{lang}')
        results = []
        for index, repo in enumerate(repositories):
            if index >= rank:
                break
            results.append(
                {
                    'index': -1,
                    'name': repo.full_name,
                    'link': repo.html_url,
                    'stars': repo.stargazers_count,
                    'description': repo.description,
                    'metric': 'GitHub',
                    'value': -1
                }
            )
        results += [{
                    'index': -1,
                    'name': 'Placeholder',
                    'link': '#empty',
                    'stars': 0,
                    'description': 'Description',
                    'metric': 'GitHub',
                    'value': -1
                }] * max(0, rank - len(results)) 
        return results, 0

def get_testcases(
    filename: str
) -> Dict[str, List[TestCase]]:
    with open(filename, 'r') as f:
        testcases = json.load(f)
    return testcases

def compute_MAP(
    relevance_sequence: List[int]
) -> float:
    precision_list = list()
    relevance_cnt = 0
    for i in range(len(relevance_sequence)):
        if relevance_sequence[i] == 0:
            pass
        else:
            relevance_cnt += 1
            precision_list.append(relevance_cnt / (i + 1))
    return sum(precision_list) / len(relevance_sequence)

def compare_searches(
    baseline_searcher: GitHubSearcher, 
    magi_indexer: MagiIndexer, 
    test_file: str = './datafile/queries.txt', 
    langs: list = ['Python'], 
    rank: int = 10, 
    get_baseline: bool = True
) -> Tuple[List[np.array], List[np.array]]:
    logger.info(f'comparing using {test_file}')
    testcases = get_testcases(test_file)
    baseline_MAPs = {lang: [] for lang in langs}
    model_MAPs = {lang: [] for lang in langs}
    for lang in langs:
        for index, (query, standard_result) in enumerate(testcases[lang]):
            model_results, _ = magi_indexer.search(query, lang=lang, rank=rank)
            model_relevance = [
                int(result['name'] in standard_result) for result in model_results
            ]
            if get_baseline:
                baseline_results, _ = baseline_searcher.search(query, lang=lang, rank=rank)
                baseline_relevance = [
                int(result['name'] in standard_result) for result in baseline_results
            ]
            else:
                baseline_relevance = [0] * len(model_relevance)
            baseline_MAPs[lang].append(compute_MAP(baseline_relevance))
            model_MAPs[lang].append(compute_MAP(model_relevance))
        baseline_MAPs[lang] = baseline_MAPs[lang]
        model_MAPs[lang] = model_MAPs[lang]
        logger.info(f'Baseline: language={lang}, mAP@{rank}={sum(baseline_MAPs[lang]) / len(baseline_MAPs[lang])}')
        logger.info(f'MAGI: language={lang}, mAP@{rank}={sum(model_MAPs[lang]) / len(model_MAPs[lang])}')
    return baseline_MAPs, model_MAPs
    
def benchmark_model(
    model, 
    corpus: str, 
    test_file: str = './datafile/queries.txt',
    embedding_file: str = None,
    langs: list = ['Python']
) -> None:
    gh = GitHubSearcher(GH_TOKEN)
    datasets = [
        GitHubCorpusRawTextDataset(corpus, lang=lang, chunk_size=1024, max_num=4) for lang in langs
    ]
    mg = MagiIndexer(datasets, model, embedding_file)
    baseline_MAPs, model_MAPs = compare_searches(gh, mg, rank=10, get_baseline=False, test_file=test_file)
    logger.info(f'baseline MAP={json.dumps(baseline_MAPs, indent=2)}, \nmodel MAP={json.dumps(model_MAPs, indent=2)}')

def cache_embeddings(
    model, 
    corpus: Union[str, list], 
    cache_loc: str, 
    langs: list = ['Python']
) -> None:
    if type(corpus) is str:
        corpus = [corpus] * len(langs)
    # print(corpus, type(corpus), type(corpus[0]))
    datasets = [
        GitHubCorpusRawTextDataset(c, lang=lang, chunk_size=1024, max_num=4) for c, lang in zip(corpus, langs)
    ]
    logger.info(f'Caching languages: {langs}, lens={[len(d) for d in datasets]}')
    mg = MagiIndexer(datasets, model)
    with open(cache_loc, 'wb') as f:
        pickle.dump(mg.embeddings, f)
    logger.info(f'Cached embeddings of {len(datasets)} datasets to {cache_loc}.')
        
def inspect_model(
    model, 
    corpus: str, 
    test_file: str = './datafile/queries.txt',
    embedding_file: str = None,
    langs: list = ['Python']
):
    datasets = [
        GitHubCorpusRawTextDataset(corpus, lang=lang, chunk_size=1024, max_num=4) for lang in langs
    ]
    mg = MagiIndexer(datasets, model, embedding_file)
    try:
        testcases = get_testcases(test_file)
    except:
        testcases = {}
    while True:
        query = input('Enter command, inspection case ID or custom query [$LANG?$QUERY]:\n')
        if query == 'q' or query == 'quit':
            print('Quit inspection.')
            break
        elif query == 's' or query == 'show':
            for lang in testcases.keys():
                for index, (query, _) in enumerate(testcases[lang]):
                    print(f'${lang} {index}. {query}')
        else:
            try:
                lang, query = query.split('?')
            except:
                continue
            try:
                query, standard_result = testcases[lang][int(query)]
            except ValueError:
                pass
            except KeyError:
                pass
            except IndexError:
                print('Index out of bound.')
                continue
            finally:
                model_results, _ = mg.search(query, lang=lang, rank=10)
                print(f'-----------------------------------\n💡  Results for "{query}" in {lang}')
                for (_, name, link, star, summary, _, score) in model_results.values():
                    print(f'➡️  {name}\n\t✏️  {summary}\n\t⭐  {star}\n\t🏅  score={score:.4f}')