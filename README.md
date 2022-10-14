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
Which includes PyTorch, transformers and other DL-related packages. We list some useful commands during development here:

Retrain:
```bash
python3 magi_models.py --train True --corpus ./datafile/ghv7_transformed.json --batch_size 16 --benchmark_file ./datafile/queries.txt 
```
Benchmark only:
```bash
python3 magi_models.py --corpus ./datafile/ghv7_transformed.json --train False --benchmark_file ./datafile/queries.txt --embedding_file ./datafile/msmarco-distilbert-base-dot-prod-v3_ghv7.pkl
```

Inspect only:
```bash
python3 magi_models.py --corpus ./datafile/ghv7_transformed.json --train False --load_from Enoch2090/MAGI --benchmark False --inspection True --benchmark_file ./datafile/queries.txt
```

Cache only:
```bash
python3 magi_models.py --corpus ./datafile/ghv7_transformed.json --train False --load_from Enoch2090/MAGI --benchmark False --inspection False --cache True --cache_loc ./datafile/msmarco-distilbert-base-dot-prod-v3_ghv7.pkl
```

Streamlit interface:
```bash
streamlit run magi.py --server.port 6006
```

Data Inspection:
```bash
streamlit run browse.py --server.port 6006
```