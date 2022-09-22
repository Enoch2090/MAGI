import random
import torch
import fire
from torch import nn
from sentence_transformers import SentenceTransformer, models, datasets, losses, InputExample
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from dataset import FineTuneDataGenerationConfig, generate_finetune_data
from indexers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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



def train(model: nn.Module, config: SentenceBertTrainConfig) -> nn.Module:
    with open('train_loss.log', 'w') as f:
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
        random.seed(42)
        random.shuffle(train_examples)
        train_dataloader = datasets.NoDuplicatesDataLoader(train_examples, batch_size=config.batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model)
        num_epochs = 3
        warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=warmup_steps, show_progress_bar=True)
    model.save(config.model_name)
    return model

def train_from_scratch(
    corpus: str=None,
    query_data: str=None,
    model_name: str=None,
    batch_size: int=16,
    num_epochs: int=3,
    benchmark: bool=True,
    cache_embeddings: bool=False
):
    assert corpus or query_data, 'ERROR: must input one of corpus or query data tsv'
    date_str = datetime.now().strftime('%y-%m-%d_%h-%M')
    model_dir = f'./datafile/{date_str}'
    if corpus:
        # regenerate the query data tsv for training
        query_data = f'./datafile/{date_str}_queries.tsv'
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
        
    model = get_distilbert_base_dotprod()
    train(model, config)
    if benchmark:
        benchmark_model(model=model, corpus=corpus)
    if cache_embeddings:
        cache_embeddings(
            model=model, 
            corpus=corpus,
            cache_loc=f'./datafile/{date_str}.npy'
        )

if __name__ == '__main__':
    fire.Fire(train_from_scratch)