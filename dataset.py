import json
import re
import os
import numpy as np
import logging
import hashlib
from magi_dataset import GitHubDataset
from dataclasses import dataclass, asdict
from abc import ABC
from tqdm.auto import tqdm
from itertools import chain

try:
    import torch
    from torch.utils.data import Dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ModuleNotFoundError:
    class Dataset(ABC): pass
    device = 'Huggingface inference API'
    
@dataclass
class FineTuneDataGenerationConfig:
    batch_size: int = 16            # Batch size
    num_queries: int = 4            # Number of queries to generate for every paragraph
    max_length_paragraph: int = 512 # Max length for paragraph
    max_length_query: int = 64      # Max length for output query
    max_chunk_per_repo: int = 5     # Max chunks selected for each repo
  
def remove_punkt(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text)

def remove_non_ascii(text: str) -> str:
    return ''.join(i for i in text if ord(i) < 128)

    
class GitHubCorpusRawTextDataset(Dataset):
    def __init__(
        self, 
        file_dir: str = None, 
        keys_used: list = ['hn_comments', 'readme'], 
        lang: str = None, 
        chunk_size:int = 512, 
        max_num:int = 10
    ):
        '''
        Arguments:
            - file_dir (str): File name of a .json file corpus. Set it as 'default' or None to let magi_dataset download the default file.
            - key_used (list[str]): A list of keys in the json file to use as corpus. 
        '''
        # with open(file_dir, 'r') as f:
        #     raw_data = json.load(f)
        if file_dir == 'default':
            file_dir = None
        raw_data = GitHubDataset(
            empty = False,
            file_path = file_dir
        )
        self.data = []
        self.raw_data = []
        self.chunk_size = chunk_size
        if lang:
            self.lang = lang
        for repo in raw_data:
            # transform the dataclass to a dictionary
            # https://docs.python.org/3/library/dataclasses.html#dataclasses.asdict
            repo = asdict(repo) 
            if lang and repo['lang'] != lang:
                # if language parameter is used during initialization,
                # initialize this dataset object as specified language only
                # this allows us to use this dataset object to retrieve information (local mode only)
                continue
            cleaned_corpus = ''
            for key in keys_used:
                cleaned_corpus += remove_non_ascii(repo[key])
                del repo[key]
            chunked_corpus = [
                cleaned_corpus[i * chunk_size: (i + 1) * chunk_size] for i in range(len(cleaned_corpus) // chunk_size)
            ]
            if len(chunked_corpus) == 0:
                continue
            repo['data'] = chunked_corpus
            repo['size'] = len(repo['data'])
            if repo['size'] <= 1:
                continue
            self.raw_data.append(repo)
        self.raw_size = len(self.raw_data)
        self.vec_to_repo = []
        self.repo_to_vec = {}
        data_index = 0
        for repo_index, repo in enumerate(self.raw_data):
            for anchor_index, anchor in enumerate(repo['data']):
                if max_num > 0 and anchor_index >=max_num:
                        break # only store the first $max_num vectors
                self.data.append(
                    (anchor, anchor_index, repo_index)
                )
                self.vec_to_repo.append(repo_index)
                try:
                    self.repo_to_vec[repo['name']].append(data_index)
                except:
                    self.repo_to_vec[repo['name']] = [data_index]
                data_index += 1
        for k, v in self.repo_to_vec.items():
            self.repo_to_vec[k] = np.array(v)
        self.size = len(self.data)
        self.repo_num = len(self.repo_to_vec.keys())

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index:int):
        return self.data[index]
    
    def get_repo(self, index:int=None, repo_index:int=None):
        if repo_index is None:
            repo_index = self.vec_to_repo[index]
        repo_data = self.raw_data[repo_index]
        return {
            'name': repo_data['name'],
            'link': repo_data['link'],
            'stars': repo_data['stars'],
            'description': repo_data['description']
        }
    
    def get_tags(self, index:int=None, repo_index:int=None):
        if repo_index is None:
            repo_index = self.vec_to_repo[index]
        repo_data = self.raw_data[repo_index]
        return [
            repo_data['tags']
        ]
    
    def find_repo(self, repo_name: str) -> int:
        try:
            return self.repo_to_vec[repo_name]
        except KeyError:
            print(f'Repository {repo_name} not found')
            return None

def get_hash(file_dir: str) -> str:
    with open(file_dir, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()
    
def generate_finetune_data(
    file_dir: str = './datafile/ghv6.json', 
    output_dir: str = 'generated_queries_all_ghv6.tsv',
    query_source: str = None,
    query_data_dir: str = './data_collection/data/data.txt'
):
    # query_source: The source of the query (stackoverflow)
    # query_data_dir: The directory of the query data
    ft_conf = FineTuneDataGenerationConfig()
    ft_dataset = GitHubCorpusRawTextDataset(file_dir, chunk_size=512, max_num=ft_conf.max_chunk_per_repo)
    ft_paragraphs = [x[0] for x in ft_dataset]
    if query_source == "stackoverflow":
        # Use the query from Stackoverflow

        out_file = open(query_data_dir, "r")
        query_data = json.loads(out_file.read())
        python_query_data = query_data['Python']
        # Currently just use python, expand to other languages later
        with open(output_dir, 'w') as f:
            for idx in tqdm(range(0, len(python_query_data))):
                query = python_query_data[idx][0]
                repo_name = python_query_data[idx][1][0]
                repo_index = ft_dataset.repo_to_vec[repo_name]
                for i in repo_index:
                    para = ft_paragraphs[i]
                    para = remove_non_ascii(para).replace("\t", " ").strip()
                    f.write("{}\t{}\n".format(query, para))

    else:
        from sentence_transformers import InputExample
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        # Decouple the transformers and PyTorch dependency
        # This allows access to the GitHubCorpusRawTextDataset object without installing PyTorch dependency
        
        ft_tokenizer = T5Tokenizer.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
        ft_model = T5ForConditionalGeneration.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
        # https://huggingface.co/BeIR/query-gen-msmarco-t5-large-v1
        ft_model.eval()
        ft_model.to(device)
        
        with open(output_dir, 'w') as f:
            for start_idx in tqdm(range(0, len(ft_paragraphs), ft_conf.batch_size)):
                sub_paragraphs = ft_paragraphs[start_idx:(start_idx + ft_conf.batch_size)]
                input_ids = ft_tokenizer.encode(
                    sub_paragraphs, 
                    max_length = ft_conf.max_length_paragraph, 
                    truncation = True, 
                    return_tensors = 'pt'
                ).to(device)
                outputs = ft_model.generate(
                    input_ids = input_ids,
                    max_length = ft_conf.max_length_query,
                    do_sample = True,
                    top_p = 0.95,
                    num_return_sequences = ft_conf.num_queries
                )
                for idx, out in enumerate(outputs):
                    query = ft_tokenizer.decode(out, skip_special_tokens = True)
                    query = remove_non_ascii(query).replace("\t", " ").strip()
                    para = sub_paragraphs[int(idx/ft_conf.num_queries)]
                    para = remove_non_ascii(para).replace("\t", " ").strip()
                    if len(query) > 0 and len(para) > 0:
                        f.write("{}\t{}\n".format(query, para))
                
if __name__ == '__main__':
    generate_finetune_data()