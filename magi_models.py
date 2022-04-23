import random
import torch
from torch import nn
from sentence_transformers import SentenceTransformer, models, datasets, losses
from dataclasses import dataclass
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass(frozen=True)
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

if __name__ == '__main__':
    config = SentenceBertTrainConfig()
    model = get_distilbert_base_dotprod()
    train(model, config)