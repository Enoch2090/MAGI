import random
import torch
import fire
import logging
from torch import nn
from sentence_transformers import SentenceTransformer, models, datasets, losses, InputExample
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from dataset import FineTuneDataGenerationConfig, generate_finetune_data
from indexers import benchmark_model, inspect_model, cache_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

@dataclass
class SentenceBertTrainConfig:
    batch_size: int = 16   # Batch size
    num_epochs: int = 3
    model_name: str = './datafile/ghv6-model'
    train_data: str = './datafile/generated_queries_all_ghv6.tsv'

def get_distilbert_base_dotprod(model_file=None):
    if model_file is not None:
        try:
            model = SentenceTransformer(model_file)
        except:
            model = SentenceTransformer('Enoch2090/MAGI')
    else:
        word_emb = models.Transformer('sentence-transformers/msmarco-distilbert-base-dot-prod-v3')
        pooling = models.Pooling(word_emb.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_emb, pooling])
    return model.to(device)



def train_model(model: nn.Module, config: SentenceBertTrainConfig) -> nn.Module:
    with open(Path(config.model_name)/'train_loss.log', 'w') as f:
        def write_score_callback(score, epoch, steps, f=f):
            f.write(f'epoch={epoch:2d}, steps={steps:6d}, score={score:.4f}')
        train_examples = [] 
        with open(config.train_data) as f:
            for line in f:
                try:
                    query, paragraph = line.strip().split('\t', maxsplit=1)
                    train_examples.append(InputExample(texts=[query, paragraph]))
                except:
                    pass
        logger.info(f'Using {len(train_examples)} lines of total train data.')
        random.seed(42)
        random.shuffle(train_examples)
        train_dataloader = datasets.NoDuplicatesDataLoader(train_examples, batch_size=config.batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model)
        num_epochs = 3
        warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=warmup_steps, show_progress_bar=True)
    model.save(config.model_name)
    logger.info(f'Model saved to {config.model_name}')
    return model

def entry( 
    train: bool = True,
    corpus: str = None,
    langs: list = ['Python', 'JavaScript'],
    query_data: str = None,
    model_name: str = None,
    batch_size: int = 16,
    num_epochs: int = 3,
    embedding_file = None,
    benchmark: bool = True,
    benchmark_file: str = None,
    inspection: bool = True,
    cache: bool = False,
    cache_loc: str = None,
    load_from: str = None
):
    if train:
        assert corpus or query_data, 'ERROR: must input one of corpus or query data tsv'
        model_name = datetime.now().strftime('%y-%m-%d_%h-%M')
        if not model_name:
            model_name = datetime.now().strftime('%y-%m-%d_%h-%M')
        model_dir = f'./datafile/{model_name}'
        if corpus:
            # regenerate the query data tsv for training
            logger.info(f'Using {corpus} to generate queries')
            query_data = f'./datafile/{model_name}_queries.tsv'
            generate_finetune_data(
                file_dir=corpus,
                output_dir=query_data
            )

        config = SentenceBertTrainConfig(
            batch_size=batch_size,
            num_epochs=num_epochs,
            model_name=model_dir,
            train_data=query_data
        )

        if not Path(model_dir).exists():
            Path(model_dir).mkdir()
            logger.info(f'Directory {model_dir} created')

        logger.info(f'Training using {query_data} with batch_size={batch_size}, num_epochs={num_epochs}')

        model = get_distilbert_base_dotprod()
        train_model(model, config)
    else:
        if load_from:
            model = get_distilbert_base_dotprod(load_from)
        else:
            model = get_distilbert_base_dotprod('Enoch2090/MAGI')
        
    if benchmark:
        logger.info(f'Benchmarking on {corpus}')
        benchmark_model(
            model=model, 
            corpus=corpus, 
            langs=langs, 
            test_file=benchmark_file,
            embedding_file=embedding_file
        )
    if inspection:
        logger.info(f'Inspection on {corpus}')
        inspect_model(
            model=model, 
            corpus=corpus, 
            langs=langs, 
            test_file=benchmark_file,
            embedding_file=embedding_file
        )
    if cache:
        if not cache_loc:
            cache_loc = f'./datafile/{model_name}.pkl'
        logger.info(f'Caching embeddings to {cache_loc}')
        cache_embeddings(
            model=model, 
            corpus=corpus,
            langs=langs, 
            cache_loc=cache_loc
        )

if __name__ == '__main__':
    fire.Fire(entry)