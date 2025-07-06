# embedding_generator.py

import pandas as pd
import torch
import numpy as np

EMBED_DIM = 100
EMBED_SAVE_PATH = "embeddings.pt"
TRIPLE_PATH = "triples.csv"

def build_vocab(df):
    entities = sorted(set(df['head']) | set(df['tail']))
    relations = sorted(set(df['relation']))
    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}
    return entity2id, relation2id

def initialize_embeddings(entity2id, relation2id, embed_dim):
    entity_embed = torch.nn.Embedding(len(entity2id), embed_dim)
    relation_embed = torch.nn.Embedding(len(relation2id), embed_dim)

    torch.nn.init.xavier_uniform_(entity_embed.weight)
    torch.nn.init.xavier_uniform_(relation_embed.weight)

    return entity_embed.weight.data, relation_embed.weight.data

def save_embeddings(entity_embed, relation_embed, entity2id, relation2id):
    torch.save({
        'entity_embeddings': entity_embed,
        'relation_embeddings': relation_embed,
        'entity2id': entity2id,
        'relation2id': relation2id
    }, EMBED_SAVE_PATH)

def main():
    df = pd.read_csv(TRIPLE_PATH)
    entity2id, relation2id = build_vocab(df)
    entity_embed, relation_embed = initialize_embeddings(entity2id, relation2id, EMBED_DIM)
    save_embeddings(entity_embed, relation_embed, entity2id, relation2id)
    print(f"Saved {len(entity2id)} entity and {len(relation2id)} relation embeddings to {EMBED_SAVE_PATH}")

if __name__ == "__main__":
    main()
