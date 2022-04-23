from magi_models import get_distilbert_base_dotprod
from dataset import GitHubCorpusRawTextDataset
from typing import List, Union, Tuple
from github import Github
from sentence_transformers import util
import numpy as np
import time
import json
import os

if "JPY_PARENT_PID" in os.environ:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

class RepoName(str): pass
class RepoDescription(str): pass
class RepoLink(str): pass
class RepoStars(int): pass
class SearchScore(int): pass
GithubQueryResult = Tuple[RepoName, RepoLink, RepoStars, RepoDescription]
TestCase = Tuple[str, List[RepoName]]

GH_TOKEN = os.getenv('GH_TOKEN') # see https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token

class MagiIndexer:
    def __init__(self, dataset, model, embedding_file=None) -> Tuple[List[GithubQueryResult], int]:
        self.dataset = dataset
        self.model = model
        if embedding_file is not None:
            with open(embedding_file, 'rb') as f:
                self.embeddings = np.load(f)
        else:
            self.embeddings = self.model.encode([x[0] for x in self.dataset])
        _, d = self.embeddings.shape
        self.pooled_embeddings = np.zeros(
            (len(self.dataset.repo_to_vec.keys()), d)
        )
        for index, (repo, repo_index) in enumerate(self.dataset.repo_to_vec.items()):
            repo_pooling = self.embeddings[repo_index].mean(axis=0)
            self.pooled_embeddings[index, :] = repo_pooling
        l, d = self.pooled_embeddings.shape
        print(f'total {l} vectors, d = {d}')
        self.pooled_embeddings = self.pooled_embeddings.astype(np.float32)
    
    def get_repo(self, index):
        return self.dataset.get_repo(repo_index=index)

    def search(self, query, rank=10):
        start = time.time()
        query_embedding = self.model.encode([query], show_progress_bar=False)
        similarity = util.dot_score(query_embedding, self.pooled_embeddings).detach().numpy().squeeze(axis=0)
        unsorted_index = np.argpartition(similarity, -rank)[-rank:]
        sorted_index = unsorted_index[np.flip(np.argsort(similarity[unsorted_index]))]
        results = []
        for index in sorted_index:
            results.append(
                tuple(
                    self.get_repo(index) + [similarity[index]]
                )
            )
        end = time.time()
        runtime = float(end - start)
        return results, runtime

class GitHubSearcher:
    def __init__(self, token: str):
        self.github_client = Github(token)
    def search(self, query, rank=10) -> Tuple[List[GithubQueryResult], int]:
        repositories = self.github_client.search_repositories(query=f'{query} stars:>10 language:Python')
        results = []
        for index, repo in enumerate(repositories):
            if index >= rank:
                break
            results.append(
                (
                    repo.full_name,
                    repo.html_url,
                    repo.stargazers_count,
                    repo.description,
                    0
                )
            )
        results += [('placeholder', '', '', 0)] * max(0, rank - len(results)) 
        return results, 0

def get_testcases(filename) -> List[TestCase]:
    with open(filename, 'r') as f:
        testcases = json.load(f)
    return testcases

def compute_MAP(relevance_sequence: List[int]):
    precision_list = list()
    relevance_cnt = 0
    for i in range(len(relevance_sequence)):
        if relevance_sequence[i] == 0:
            pass
        else:
            relevance_cnt += 1
            precision_list.append(relevance_cnt / (i + 1))
    return sum(precision_list) / len(relevance_sequence)

def compare_searches(baseline_searcher: GitHubSearcher, magi_indexer: MagiIndexer, test_file='./datafile/queries.txt', rank=10, get_baseline=True):
    testcases = get_testcases(test_file)
    baseline_MAPs = []
    model_MAPs = []
    for index, (query, standard_result) in enumerate(testcases):
        model_results, _ = magi_indexer.search(query, rank=rank)
        model_relevance = [
            int(result[0] in standard_result) for result in model_results
        ]
        if get_baseline:
            baseline_results, _ = baseline_searcher.search(query, rank=rank)
            baseline_relevance = [
            int(result[0] in standard_result) for result in baseline_results
        ]
        else:
            baseline_relevance = [0] * len(model_relevance)
        
        baseline_MAPs.append(compute_MAP(baseline_relevance))
        model_MAPs.append(compute_MAP(model_relevance))
    baseline_MAPs = np.array(baseline_MAPs)
    model_MAPs = np.array(model_MAPs)
    print(f'Baseline: mAP@{rank}={baseline_MAPs.mean()}')
    print(f'MAGI: mAP@{rank}={model_MAPs.mean()}')
    return baseline_MAPs, model_MAPs
    
if __name__=='__main__':
    model = get_distilbert_base_dotprod('./datafile/ghv5-model')
    gh = GitHubSearcher(GH_TOKEN)
    dataset = GitHubCorpusRawTextDataset('./datafile/ghv6.json', mode='index', chunk_size=1024, max_num=4)
    mg = MagiIndexer(dataset, model)
#             baseline_MAPs, model_MAPs = compare_searches(gh, mg, rank=5, get_baseline=False)

    baseline_MAPs, model_MAPs = compare_searches(gh, mg, rank=10, get_baseline=False)
    print(mg.search('extract articles from web pages'))
    with open('msmarco-distilbert-base-dot-prod-v3_trained_embeddings.npy', 'wb') as f:
        np.save(f, mg.embeddings)