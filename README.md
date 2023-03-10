# MAGI

<img src="https://raw.githubusercontent.com/Enoch2090/MAGI/main/resources/MAGI_title.001.jpeg" alt="MAGI_title.001" style="zoom: 33%;" />

## What is MAGI?

Read [**MAGI: MachineLearning Augmented Github Indexer**](https://www.enoch2090.me/article/MAGI-MachineLearning-Augmented-Github-Indexer) for more information.

## How to use it?

Demo at: [magi-search.com](https://magi-search.com).

To use MAGI, simply type in description of the package you are looking for. The more detailed your description is, the more accurate the result will be.

## Deployment

You can find the tuned model at [https://huggingface.co/Enoch2090/MAGI](https://huggingface.co/Enoch2090/MAGI).
Current version of corpus is located at [https://huggingface.co/datasets/Enoch2090/github_semantic_search/raw/main/ghv7_transformed.json](https://huggingface.co/datasets/Enoch2090/github_semantic_search/raw/main/ghv7_transformed.json).

Ubuntu is recommended to deploy MAGI. Other Linux distributions & Windows should also work (not tested though). To deploy the streamlit dashboard, first set in the terminal:
```shell
export HUGGINGFACE_TOKEN="Bearer YOUR_TOKEN"
```
Then use `requirements.txt` to install the requirements:
```bash
pip install -r requirements.txt
```
## Development

The development enviroment is installed via:
```bash
pip install -r requirements_dev.txt
```
Which includes PyTorch, transformers and other DL-related packages. We list some useful commands during development here. 

Retrain:
```bash
python3 magi_models.py --train True --corpus default --batch_size 16 --benchmark True --benchmark_file ./datafile/queries.txt --inspection False
```

By setting `corpus='default'`, we use the default data pulled according to the list https://huggingface.co/datasets/Enoch2090/github_semantic_search/blob/main/list.json. You may also set this parameter as a valid .json file.

Benchmark only:
```bash
python3 magi_models.py --corpus default --train False --benchmark_file ./datafile/queries.txt --embedding_file ./datafile/msmarco-distilbert-base-dot-prod-v3_ghv7.pkl
```

Inspect only:
```bash
python3 magi_models.py --corpus default --train False --load_from Enoch2090/MAGI --benchmark False --inspection True --benchmark_file ./datafile/queries.txt
```
This mode is used to inspect the efficiency of models via the mAP metric, given the query file `./datafile/queries.txt`.

Cache only:
```bash
python3 magi_models.py --corpus "[\"python-latest\",\"javascript-latest\",\"cpp-latest\",\"rust-latest\",\"go-latest\"]" --langs [Python,JavaScript,\"C++\",Rust,Go] --train False --load_from Enoch2090/MAGI --benchmark False --inspection False --cache True --cache_loc ./datafile/MAGI_ghv10.pkl
```
Or the defaults

```bash
python3 magi_models.py --train False --load_from Enoch2090/MAGI --benchmark False --inspection False --cache True --cache_loc ./datafile/MAGI_ghv10.pkl
```



This mode is used when training is complete. Use this mode to convert the database into embeddings and cache into a .pkl file.

Streamlit interface:
```bash
streamlit run magi.py --server.port 6006
```
This script provides a simple user interface via Streamlit. Not intended for production.

Data Inspection:
```bash
streamlit run browse.py --server.port 6006
```
This script uses fuzzy search to match exact repo names, allowing developers to check whether an exact repo is in the database, and to inspect the raw data of that repo.

## Model Design Choices
Current architecture:
- Chunkify corpus in each repository to 512 words chunks (`dataset.GitHubCorpusRawTextDataset`).
- Use T5 model to generate synthetic queries on the first few chunks of each repository. The underlying idea is that the first few chunks in each repository should have more introductions on its use. Note that in the `__init__` method of `dataset.GitHubCorpusRawTextDataset`, the parameter keys_used defaults to `['hn_comments', 'readme']`. Corpus value stored in these keys are merged into one single string first before the chunk process, therefore if any HackerNews comments exist, they will appear before the GitHub README as I identify them as more valuable corpus. This design may be changed if other choices yields better results.
- After the synthetic queries are generated (`dataset.generate_finetune_data`), train on the (corpus, query) tuples to finetune the Sentence Transformer.
- Use the finetuned transformer to encode the database into embeddings. In `indexers.cache_embeddings`, you may find that currently only the first 4 chunks for each repository is cached into embeddings. That means for each repo, it has an embedding of the shape (n, 768) where n is 1, 2, 3 or 4.
- Use the finetuned transformer to encode query. In `indexers.MagiIndexer`, the encoded query is compared with stored embeddings for the selected programming language.

Future Works:
- Identify in each repo which chunks are more valuable to keep, instead of brutely keeping the first few. 
- Use StackOverFlow API and HackerNews API to mine query-result pairs.
- Use the previous result, curate a list of query-result pairs for each language as benchmark standard. We only have Python at the moment, and the number of queries is small.
- Use the previous result, introduce the query-result pairs in the training process to further finetune the model.

## Related Projects
- [Enoch2090/magi_dataset](https://github.com/Enoch2090/magi_dataset): Data interface for MAGI.
- [Grep.app](https://grep.app/): Exact match search over GitHub.
- [arXiv Xplorer](https://arxivxplorer.com): Semantic search on arXiv for finding papers.