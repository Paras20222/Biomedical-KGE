import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from tqdm import tqdm
import random
import argparse
import sys

class KGDataset(Dataset):
    def __init__(self, triple_file, entity_to_id, relation_to_id):
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id
        self.num_entities = len(entity_to_id)
        self.triples = []
        with open(triple_file, 'r', encoding='utf-8') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                if h in entity_to_id and r in relation_to_id and t in entity_to_id:
                    self.triples.append((entity_to_id[h], relation_to_id[r], entity_to_id[t]))
        self.triples = np.array(self.triples, dtype=np.int64)
        self.positive_set = set(map(tuple, self.triples))
    
    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        # Create positive triple tensor
        pos = torch.tensor([h, r, t], dtype=torch.long)
        neg_triples = []
        # Create negative triple by corrupting head
        neg_h = random.randint(0, self.num_entities - 1)
        while (neg_h, r, t) in self.positive_set:
            neg_h = random.randint(0, self.num_entities - 1)
        neg_triples.append([neg_h, r, t])
        # Create negative triple by corrupting tail
        neg_t = random.randint(0, self.num_entities - 1)
        while (h, r, neg_t) in self.positive_set:
            neg_t = random.randint(0, self.num_entities - 1)
        neg_triples.append([h, r, neg_t])
        # Combine all triples with corresponding labels
        neg_triples = torch.tensor(neg_triples, dtype=torch.long)
        labels = torch.tensor([1, -1, -1], dtype=torch.float)
        all_triples = torch.cat([pos.unsqueeze(0), neg_triples], dim=0)
        return all_triples, labels

def collate_fn(batch):
    triples = torch.cat([item[0] for item in batch])
    labels = torch.cat([item[1] for item in batch])
    return triples, labels

class AdvancedConvBlock(nn.Module):
    """Handles asymmetric padding for even kernel sizes with proper dimension handling"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        
        # Since input sequence has length 3, we need to be careful with kernel sizes
        # For a sequence of length 3, max kernel size should be 3
        self.effective_kernel = min(kernel_size, 3)
        
        # Calculate padding based on effective kernel size
        self.padding = (self.effective_kernel - 1) // 2
        
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, self.effective_kernel,
            padding=self.padding, groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        
        # Residual connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
            
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = self.shortcut(x)
        
        # Apply depthwise convolution
        out = F.gelu(self.depthwise(x))
        out = self.pointwise(out)
        
        # Ensure output and residual have same shape
        if out.shape != residual.shape:
            # If shapes don't match, use adaptive pooling to match dimensions
            if out.shape[-1] != residual.shape[-1]:
                out = F.adaptive_max_pool1d(out, residual.shape[-1])
        
        out = self.bn(out)
        out = self.dropout(out)
        out += residual
        return F.gelu(out)

class KGEModel(nn.Module):
    def __init__(self, entity_embeddings, relation_embeddings, 
                 filter_sizes=(3, 4, 5), num_filters=128, dropout=0.3, 
                 l2_lambda=1e-5, num_conv_layers=3):
        super().__init__()
        self.embed_dim = entity_embeddings.shape[1]
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_lambda = l2_lambda

        # Embedding layers
        self.entity_embed = nn.Embedding.from_pretrained(
            torch.FloatTensor(entity_embeddings), freeze=False
        )
        self.rel_embed = nn.Embedding.from_pretrained(
            torch.FloatTensor(relation_embeddings), freeze=False
        )

        # Convolutional branches with dimension preservation
        self.conv_branches = nn.ModuleList()
        for fs in filter_sizes:
            branch = nn.Sequential()
            in_channels = self.embed_dim
            for layer_idx in range(num_conv_layers):
                out_channels = num_filters if layer_idx == num_conv_layers-1 else num_filters//2
                branch.add_module(
                    f"conv_{fs}_{layer_idx}",
                    AdvancedConvBlock(
                        in_channels,
                        out_channels,
                        kernel_size=fs
                    )
                )
                in_channels = out_channels
            branch.add_module(f"pool_{fs}", nn.AdaptiveMaxPool1d(1))
            self.conv_branches.append(branch)

        # Final dense layers
        self.fc = nn.Sequential(
            nn.Linear(len(filter_sizes)*num_filters, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        h, r, t = x[:, 0], x[:, 1], x[:, 2]
        h_emb = self.entity_embed(h)
        r_emb = self.rel_embed(r)
        t_emb = self.entity_embed(t)
        
        # Correct input shape: [batch, embed_dim, 3]
        triple_emb = torch.stack([h_emb, r_emb, t_emb], dim=1).permute(0, 2, 1)
        
        # Process through branches
        branch_outputs = []
        for branch in self.conv_branches:
            branch_out = branch(triple_emb)
            branch_outputs.append(branch_out.squeeze(-1))
            
        combined = torch.cat(branch_outputs, dim=1)
        return self.fc(combined).squeeze()

    def l2_regularization(self):
        return self.l2_lambda * sum(p.norm(2)**2 for p in self.parameters())

def kged_loss(scores, labels, model):
    loss = torch.log(1 + torch.exp(-labels * scores)).mean()
    return loss + model.l2_regularization()

def train_model(model, train_loader, val_loader, optimizer, device, epochs=10, patience=3):
    model.to(device)
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', mininterval=10)
        for batch_idx, (triples, labels) in enumerate(pbar):
            triples = triples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            scores = model(triples)
            loss = kged_loss(scores, labels, model)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (scores >= 0).float() * 2 - 1
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.4f}',
                'Acc': f'{correct/total:.4f}'
            })
        
        val_acc = evaluate(model, val_loader, device)
        print(f'Validation - Accuracy: {val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), '/kaggle/working/best_model.pt')
            print(f'Model saved to /kaggle/working/best_model.pt')
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter} epochs')
        
        if patience_counter >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for triples, labels in tqdm(loader, desc='Evaluating'):
            triples = triples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            scores = model(triples)
            predicted = (scores >= 0).float() * 2 - 1
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f'Evaluation - Accuracy: {accuracy:.4f}')
    return accuracy

def inspect_checkpoint(checkpoint_path, device):
    """Examine the architecture of the saved model"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Analyze keys to determine model architecture
        filter_sizes = set()
        num_filters = None
        num_conv_layers = 0
        
        for key in checkpoint.keys():
            if 'conv_branches' in key:
                parts = key.split('.')
                if len(parts) >= 3 and 'conv_' in parts[2]:
                    filter_info = parts[2].split('_')
                    if len(filter_info) >= 2 and filter_info[1].isdigit():
                        filter_size = int(filter_info[1])
                        filter_sizes.add(filter_size)
                        
                        if len(filter_info) >= 3 and filter_info[2].isdigit():
                            layer_idx = int(filter_info[2]) + 1
                            num_conv_layers = max(num_conv_layers, layer_idx)
            
            if 'pointwise.weight' in key:
                shape = checkpoint[key].shape
                if len(shape) > 0:
                    num_filters = max(num_filters or 0, shape[0])
        
        # Try FC layer if we couldn't determine filters
        if num_filters is None and 'fc.0.weight' in checkpoint:
            fc_shape = checkpoint['fc.0.weight'].shape
            if len(filter_sizes) > 0:
                num_filters = fc_shape[1] // len(filter_sizes)
                
        return {
            'filter_sizes': tuple(sorted(filter_sizes)) if filter_sizes else (3, 4, 5),
            'num_filters': num_filters if num_filters else 128,
            'num_conv_layers': num_conv_layers if num_conv_layers > 0 else 3
        }
        
    except Exception as e:
        print(f"Error inspecting checkpoint: {e}")
        return {
            'filter_sizes': (3, 4, 5),
            'num_filters': 128,
            'num_conv_layers': 3
        }

def load_model_with_compatible_weights(model, checkpoint_path, device):
    """Load a model checkpoint, handling parameter mismatches"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get the model's state dict
        model_dict = model.state_dict()
        
        # Create a filtered state dict with only compatible parameters
        filtered_dict = {}
        skipped_params = []
        
        for k, v in checkpoint.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    skipped_params.append((k, v.shape, model_dict[k].shape))
        
        # Load the filtered state dict
        model.load_state_dict(filtered_dict, strict=False)
        
        print(f"Successfully loaded {len(filtered_dict)}/{len(model_dict)} parameters")
        print(f"Skipped {len(skipped_params)} parameters due to mismatch")
        
        return model
    
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return model

def compute_ranking_metrics(model, test_triples, all_triples, entity_to_id, device, sample_size=4000):
    model.eval()
    num_entities = len(entity_to_id)
    entity_ids = torch.arange(num_entities).to(device)
    
    # Sample test triples if needed
    test_triples = test_triples[:sample_size]
    print(f"Loaded {len(test_triples)} test triples for evaluation")
    
    # Create filtered dictionary for evaluation
    filter_dict = {}
    for h, r, t in all_triples:
        filter_dict.setdefault((r, t), set()).add(h)
        filter_dict.setdefault((h, r), set()).add(t)
    
    ranks = []
    hits = {1: 0, 3: 0, 10: 0}
    
    # Process in batches to avoid OOM
    batch_size = 128
    
    with torch.no_grad():
        for i, (h, r, t) in enumerate(tqdm(test_triples, desc="Evaluation")):
            # Head prediction
            head_ranks = []
            
            for start in range(0, num_entities, batch_size):
                end = min(start + batch_size, num_entities)
                current_entities = entity_ids[start:end]
                
                # Create batch for head prediction
                hr_batch = torch.zeros((len(current_entities), 3), dtype=torch.long, device=device)
                hr_batch[:, 0] = current_entities
                hr_batch[:, 1] = r
                hr_batch[:, 2] = t
                
                # Get scores
                scores = model(hr_batch).cpu().numpy()
                
                # Filter out other true triples
                for j, e in enumerate(current_entities.cpu().numpy()):
                    if e != h and (r, t) in filter_dict and e in filter_dict[(r, t)]:
                        scores[j] = -np.inf
                
                # If true head is in this batch
                if start <= h < end:
                    h_idx = h - start
                    h_score = scores[h_idx]
                    h_rank = 1 + np.sum(scores > h_score)
                    head_ranks.append(h_rank)
            
            if head_ranks:
                ranks.append(min(head_ranks))
            
            # Tail prediction
            tail_ranks = []
            
            for start in range(0, num_entities, batch_size):
                end = min(start + batch_size, num_entities)
                current_entities = entity_ids[start:end]
                
                # Create batch for tail prediction
                tr_batch = torch.zeros((len(current_entities), 3), dtype=torch.long, device=device)
                tr_batch[:, 0] = h
                tr_batch[:, 1] = r
                tr_batch[:, 2] = current_entities
                
                # Get scores
                scores = model(tr_batch).cpu().numpy()
                
                # Filter out other true triples
                for j, e in enumerate(current_entities.cpu().numpy()):
                    if e != t and (h, r) in filter_dict and e in filter_dict[(h, r)]:
                        scores[j] = -np.inf
                
                # If true tail is in this batch
                if start <= t < end:
                    t_idx = t - start
                    t_score = scores[t_idx]
                    t_rank = 1 + np.sum(scores > t_score)
                    tail_ranks.append(t_rank)
            
            if tail_ranks:
                ranks.append(min(tail_ranks))
    
    # Calculate metrics
    ranks_array = np.array(ranks)
    mrr = np.mean(1.0 / ranks_array)
    
    hits_metrics = {}
    for k in [1, 3, 10]:
        hits_metrics[k] = np.mean(ranks_array <= k)
    
    return {'MRR': mrr, **{f'Hits@{k}': hits_metrics[k] for k in hits_metrics}}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    base_path = '/kaggle/input/embeddings-triplet-dataset/'
    
    entity_emb = np.load(os.path.join(base_path, 'entity_embeddings_continued.npy'))
    rel_emb = np.load(os.path.join(base_path, 'relation_embeddings_continued.npy'))
    
    with open(os.path.join(base_path, 'entity_to_id.json'), 'r') as f:
        entity_to_id = json.load(f)
    with open(os.path.join(base_path, 'relation_to_id.json'), 'r') as f:
        relation_to_id = json.load(f)
    
    # Create datasets
    train_set = KGDataset(
        os.path.join(base_path, 'train.tsv'),
        entity_to_id,
        relation_to_id
    )
    val_set = KGDataset(
        os.path.join(base_path, 'val.tsv'),
        entity_to_id,
        relation_to_id
    )
    test_set = KGDataset(
        os.path.join(base_path, 'test.tsv'),
        entity_to_id,
        relation_to_id
    )
    
    batch_size = 512
    
    # Create data loaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True, persistent_workers=True
    )
    
    # Initialize model
    model = KGEModel(
        entity_embeddings=entity_emb,
        relation_embeddings=rel_emb,
        filter_sizes=(3, 4, 5),
        num_filters=128,
        dropout=0.3,
        num_conv_layers=3,
        l2_lambda=1e-5
    )
    
    # Train model
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    train_model(model, train_loader, val_loader, optimizer, device, epochs=4, patience=1)
    
    # Test model
    print("Loading best model for testing...")
    checkpoint_path = '/kaggle/working/best_model.pt'
    
    # First, inspect checkpoint to get architecture
    architecture = inspect_checkpoint(checkpoint_path, device)
    
    # Initialize model with detected architecture
    model = KGEModel(
        entity_embeddings=entity_emb,
        relation_embeddings=rel_emb,
        filter_sizes=architecture['filter_sizes'],
        num_filters=architecture['num_filters'],
        dropout=0.3,
        num_conv_layers=architecture['num_conv_layers'],
        l2_lambda=1e-5
    ).to(device)
    
    # Load weights with compatibility handling
    model = load_model_with_compatible_weights(model, checkpoint_path, device)
    
    print("Testing model...")
    evaluate(model, test_loader, device)
    
    # Load test data for ranking metrics
    test_triples = []
    with open(os.path.join(base_path, 'test.tsv'), 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            if h in entity_to_id and r in relation_to_id and t in entity_to_id:
                test_triples.append((
                    entity_to_id[h],
                    relation_to_id[r],
                    entity_to_id[t]
                ))
    
    # Load all triples for filtering
    all_triples = set()
    for split in ['train.tsv', 'val.tsv', 'test.tsv']:
        with open(os.path.join(base_path, split), 'r') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                if h in entity_to_id and r in relation_to_id and t in entity_to_id:
                    all_triples.add((
                        entity_to_id[h],
                        relation_to_id[r],
                        entity_to_id[t]
                    ))
    
    print(f"Loaded {len(all_triples)} total triples for filtering")
    
    # Compute ranking metrics
    print("Computing MRR and Hits@k ...")
    metrics = compute_ranking_metrics(
        model, test_triples, all_triples, 
        entity_to_id, device, sample_size=200
    )
    
    print("\n=== Final Test Metrics ===")
    print(f"MRR: {metrics['MRR']:.4f}")
    print(f"Hits@1: {metrics['Hits@1']:.4f}")
    print(f"Hits@3: {metrics['Hits@3']:.4f}")
    print(f"Hits@10: {metrics['Hits@10']:.4f}")

if __name__ == "__main__":
    sys.argv = [sys.argv[0]]
    main()
