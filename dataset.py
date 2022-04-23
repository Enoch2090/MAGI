import json
import re
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sentence_transformers import InputExample
from nltk.corpus import stopwords
from tqdm.notebook import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from dataclasses import dataclass

if "JPY_PARENT_PID" in os.environ:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
    
try:
    stop_words = set(stopwords.words('english'))
except:
    import nltk
    nltk.download('stopwords')
finally:
    stop_words = set(stopwords.words('english'))
   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass(frozen=True)
class FineTuneDataGenerationConfig:
    batch_size: int = 16   # Batch size
    num_queries: int = 8   # Number of queries to generate for every paragraph
    max_length_paragraph: int = 512 # Max length for paragraph
    max_length_query: int = 64   # Max length for output query

def remove_punkt(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text)

def remove_non_ascii(text: str) -> str:
    return ''.join(i for i in text if ord(i)<128)

def remove_stopwords(text_list: list) -> list:
    return [w for w in text_list if not w.lower() in stop_words]

    
class GitHubCorpusRawTextDataset(Dataset):
    def __init__(self, file_dir: str, keys_used: list = ['hn_comments', 'readme'], chunk_size:int = 512, max_num:int = 10, mode='pn'):
        '''
        Arguments:
            - file_dir (str): File name of a .json file corpus.
            - key_used (list[str]): A list of keys in the json file to use as corpus. 
        '''
        with open(file_dir, 'r') as f:
            raw_data = json.load(f)
        self.raw_data = []
        self.data = []
        self.chunk_size = chunk_size
        for repo in raw_data:
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
                if not mode == 'index':
                    pos_sample_index = anchor_index
                    neg_sample_index = 0
                    neg_repo_index = repo_index
                    while pos_sample_index == anchor_index:
                        pos_sample_index = np.random.randint(low=0, high=repo['size'])
                    while neg_repo_index == repo_index:
                        neg_repo_index = np.random.randint(low=0, high=self.raw_size)
                    neg_sample_index = np.random.randint(low=0, high=self.raw_data[neg_repo_index]['size'])

                if mode == 'triplet':
                    self.data.append(
                        (
                            anchor,
                            repo['data'][pos_sample_index],
                            self.raw_data[neg_repo_index]['data'][neg_sample_index]
                        )
                    )
                elif mode == 'pn_train':
                    self.data.append(
                        InputExample(
                            texts=[anchor, repo['data'][pos_sample_index]],
                            label=0.9
                        )
                    )
                    self.data.append(
                        InputExample(
                            texts=[anchor, self.raw_data[neg_repo_index]['data'][neg_sample_index]],
                            label=0.1
                        )
                    )
                elif mode == 'pn_dev':
                    self.data.append(
                        (anchor, repo['data'][pos_sample_index], 0.9)
                    )
                    self.data.append(
                        (anchor, self.raw_data[neg_repo_index]['data'][neg_sample_index], 0.1)
                    )
                elif mode == 'index':
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
#         print(f'Successfully built dataset, total {self.size} triplets.')

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]
    
    def get_repo(self, index=None, repo_index=None):
        if repo_index is None:
            repo_index = self.vec_to_repo[index]
        repo_data = self.raw_data[repo_index]
        return [
            repo_data['name'],
            repo_data['link'],
            repo_data['stars'],
            repo_data['description']
        ]
    
    def get_tags(self, index=None, repo_index=None):
        if repo_index is None:
            repo_index = self.vec_to_repo[index]
        repo_data = self.raw_data[repo_index]
        return [
            repo_data['tags']
        ]

def generate_finetune_data(file_dir: str='./datafile/ghv6.json', output_dir: str='generated_queries_all_ghv6.tsv'):
    ft_conf = FineTuneDataGenerationConfig()
    ft_dataset = GitHubCorpusRawTextDataset(file_dir, mode='index', chunk_size=512, max_num=50)
    ft_paragraphs = [x[0] for x in ft_dataset]
    ft_tokenizer = T5Tokenizer.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
    ft_model = T5ForConditionalGeneration.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
    ft_model.eval()
    ft_model.to(device)
    with open(output_dir, 'w') as f:
        for start_idx in tqdm(range(0, len(ft_paragraphs), ft_conf.batch_size)):
            sub_paragraphs = ft_paragraphs[start_idx:start_idx + ft_conf.batch_size]
            inputs = ft_tokenizer.prepare_seq2seq_batch(sub_paragraphs, max_length=ft_conf.max_length_paragraph, truncation=True, return_tensors='pt').to(device)
            outputs = ft_model.generate(
                **inputs,
                max_length=ft_conf.max_length_query,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=ft_conf.num_queries
            )
            for idx, out in enumerate(outputs):
                query = ft_tokenizer.decode(out, skip_special_tokens=True)
                query = remove_non_ascii(query)
                para = sub_paragraphs[int(idx/ft_conf.num_queries)]
                para = remove_non_ascii(para)
                f.write("{}\t{}\n".format(query.replace("\t", " ").strip(), para.replace("\t", " ").strip()))
                
if __name__ == '__main__':
    generate_finetune_data()