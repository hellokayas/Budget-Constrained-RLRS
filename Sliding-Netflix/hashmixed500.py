import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
from torch.utils.data import Subset
import xxhash  # Make sure to install xxhash (e.g. pip install xxhash)

# ---------------------------
# Hashing Function for Categorical Features
# ---------------------------
def hash_feature(feature_value: str, feature_name: str):
    """
    Hashes a feature value using xxhash with the feature name as seed.
    This ensures that two different features (even with the same value)
    are hashed differently.
    """
    seed = xxhash.xxh32(feature_name, 0).intdigest()
    return xxhash.xxh64(feature_value, seed).intdigest() - 2 ** 63

# ---------------------------
# Interaction Type Mapping
# ---------------------------
INTERACTION_TYPE_MAPPING = {
    "pv": 0,
    "buy": 1,
    "cart": 2,
    "fav": 3
}

# ---------------------------
# KShiftEmbedding Module
# ---------------------------
class KShiftEmbedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            emb_dim: int,
            num_shifts: int = 8,
            normalize_output: bool = True,
            sparse: bool = False,
    ):
        """
        :param num_embeddings: Number of rows in the (hashed) embedding table.
        :param emb_dim: Dimensionality of the embeddings.
        :param num_shifts: Number of bit-shift operations (more shifts gives a richer combination).
        :param normalize_output: If True, output embeddings are L2 normalized.
        :param sparse: Whether to use sparse gradients.
        """
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, emb_dim, sparse=sparse)
        self._num_embeddings = num_embeddings
        self._num_shifts = num_shifts
        self._num_bits = 64
        self._normalize_output = normalize_output

    def forward(self, id_: torch.LongTensor) -> torch.Tensor:
        """
        :param id_: A tensor of hashed IDs.
        :return: The combined embedding for each hashed ID.
        """
        tensors = []
        for col_idx in range(self._num_shifts):
            idx = self.get_row_idx(id_, col_idx)
            tensors.append(self.emb(idx))
        x = torch.stack(tensors, dim=-1).sum(dim=-1)
        if self._normalize_output:
            x = F.normalize(x, p=2.0, dim=-1)
        else:
            x = x / math.sqrt(self._num_shifts)
        return x

    def get_row_idx(self, x: torch.LongTensor, col_idx: int):
        if col_idx != 0:
            # Perform bit-shift operations (similar to bit rotation)
            x = (x << col_idx) | (x >> (self._num_bits - col_idx))
        # Map to a valid index range using modulo operation
        return torch.remainder(x, self._num_embeddings)

# ---------------------------
# Dataset and Sampling Methods
# ---------------------------
class InteractionDataset(Dataset):
    def __init__(self, data_path, mode="control", window_size=100, max_history=None, sliding_stride=1):
        colnames = ["user_id", "item_id", "category_id", "interaction_type", "timestamp"]
        # Read the CSV file with specified column names
        self.df = pd.read_csv(data_path, names=colnames, header=None)
        self.df.sort_values(["user_id", "timestamp"], inplace=True)
        print(len(self.df))
        # **Step 1: Compute item interaction frequency**
        item_counts = self.df["item_id"].value_counts()
        # **Step 2: Remove infrequent items**
        frequent_items = item_counts[item_counts >= 50].index  # Items with enough interactions
        self.df = self.df[self.df["item_id"].isin(frequent_items)]
        print(len(self.df))
        # Create a mapping from raw item IDs to contiguous indices (0-indexed)
        unique_items = sorted(self.df["item_id"].unique())
        self.item2idx = {item: idx for idx, item in enumerate(unique_items)}
        
        self.user_sequences = {}
        for user, group in self.df.groupby("user_id"):
            group = group.sort_values("timestamp")
            items = group["item_id"].tolist()
            types = group["interaction_type"].map(lambda x: INTERACTION_TYPE_MAPPING.get(x, 0)).tolist()
            self.user_sequences[user] = (items, types)
        
        self.mode = mode
        self.window_size = window_size
        self.max_history = max_history
        self.sliding_stride = sliding_stride

        self.samples = []
        for user, (items, types) in self.user_sequences.items():
            if self.mode == "control":
                if len(items) >= window_size:
                    sample_items = items[-window_size:]
                    sample_types = types[-window_size:]
                else:
                    sample_items = items
                    sample_types = types
                self.samples.append((user, sample_items, sample_types))
            elif self.mode == "sliding":
                if self.max_history is not None:
                    items = items[-self.max_history:]
                    types = types[-self.max_history:]
                if len(items) < window_size:
                    self.samples.append((user, items, types))
                else:
                    for i in range(0, len(items) - window_size + 1, self.sliding_stride):
                        window_items = items[i:i + window_size]
                        window_types = types[i:i + window_size]
                        self.samples.append((user, window_items, window_types))
            else:
                raise ValueError("Invalid mode. Use 'control' or 'sliding'.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        user, item_seq, type_seq = self.samples[idx]
        # ---- Apply hashing to each raw item_id for input features ----
        hashed_item_seq = [hash_feature(str(item), "item_id") for item in item_seq]
        # Use hashed IDs as inputs
        input_items_full = torch.tensor(hashed_item_seq, dtype=torch.long)
        # Ensure that the sequence length is at least 2; if not, pad with 0.
        if len(input_items_full) < 2:
            input_items_full = torch.cat([input_items_full, torch.tensor([0], dtype=torch.long)])
            type_seq = type_seq + [0]
        # Input sequence is all but the last element.
        input_items = input_items_full[:-1]
        # Convert target raw item IDs to contiguous indices using the mapping.
        target_items = torch.tensor([self.item2idx[item] for item in item_seq[1:]], dtype=torch.long)
        input_types = torch.tensor(type_seq[:-1], dtype=torch.long)
        return input_items, target_items, input_types

def pad_collate_fn(batch):
    """
    Custom collate function to pad sequences to the maximum length in the batch.
    Each sample is a tuple: (input_items, target_items, input_types)
    """
    input_items, target_items, input_types = zip(*batch)
    input_items_padded = nn.utils.rnn.pad_sequence(input_items, batch_first=True, padding_value=0)
    target_items_padded = nn.utils.rnn.pad_sequence(target_items, batch_first=True, padding_value=0)
    input_types_padded = nn.utils.rnn.pad_sequence(input_types, batch_first=True, padding_value=0)
    return input_items_padded, target_items_padded, input_types_padded

# ---------------------------
# Modified Model Definition: Gemma2 with K-Shift Item Embedding
# ---------------------------
class Gemma2(nn.Module):
    def __init__(self, num_items, num_interaction_types=4, emb_dim=64, n_layers=2, n_heads=4, dropout=0.1, max_seq_len=100):
        super(Gemma2, self).__init__()
        # Instead of using nn.Embedding for items, we use KShiftEmbedding.
        # We “expand” the number of embeddings slightly to allow hashing to be more expressive.
        expansion_factor = 1.15
        num_emb = int(expansion_factor * num_items)
        self.item_embedding = KShiftEmbedding(num_embeddings=num_emb, emb_dim=emb_dim, num_shifts=8, normalize_output=True)
        
        # Interaction types are few so we keep the simple embedding.
        self.interaction_embedding = nn.Embedding(num_interaction_types, emb_dim)
        self.position_embedding = nn.Embedding(max_seq_len, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # The output layer projects back to the target space (contiguous indices)
        self.output_layer = nn.Linear(emb_dim, num_items)
        self.max_seq_len = max_seq_len

    def forward(self, x, interaction_types):
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        # x contains hashed item IDs; the KShiftEmbedding will use these to fetch compressed embeddings.
        item_emb = self.item_embedding(x)
        type_emb = self.interaction_embedding(interaction_types)
        pos_emb = self.position_embedding(positions)
        x = item_emb + type_emb + pos_emb
        x = x.transpose(0, 1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        x = self.transformer(x, mask=mask)
        x = x.transpose(0, 1)
        logits = self.output_layer(x)
        return logits

# ---------------------------
# Training and Evaluation Functions
# ---------------------------
def train_model(model, dataloader, num_epochs=10, lr=0.001, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for input_items, target_items, input_types in tqdm(dataloader):
            input_items = input_items.to(device)
            target_items = target_items.to(device)
            input_types = input_types.to(device)
            optimizer.zero_grad()
            logits = model(input_items, input_types)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_items.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), "taobaocontrol.pth")
        print("Model saved as taobaocontrol.pth")

def train_model_mixed(control_dataloader, sliding_dataloader, model, num_epochs=10, X=5, lr=0.001, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(num_epochs):
        if epoch < X:
            dataloader = control_dataloader
            print(f"Epoch {epoch + 1}/{num_epochs} using CONTROL sampling")
        else:
            dataloader = sliding_dataloader
            print(f"Epoch {epoch + 1}/{num_epochs} using SLIDING sampling")
        
        epoch_loss = 0.0
        for input_items, target_items, input_types in tqdm(dataloader):
            input_items = input_items.to(device)
            target_items = target_items.to(device)
            input_types = input_types.to(device)
            optimizer.zero_grad()
            logits = model(input_items, input_types)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_items.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), "taobaomixed500.pth")
        print("Model saved as taobaomixed500.pth")

def evaluate_model(model, dataloader, device='cpu', top_k=10):
    """
    Evaluate the model by computing the average loss, perplexity, MRR, MAP, and Recall@top_k.
    For each prediction (at each time step), we compute the rank of the ground truth item.
    """
    model.eval()
    total_loss = 0.0
    total_mrr = 0.0
    total_recall = 0.0
    total_count = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for input_items, target_items, input_types in tqdm(dataloader):
            input_items = input_items.to(device)
            target_items = target_items.to(device)
            input_types = input_types.to(device)
            logits = model(input_items, input_types)  # Shape: (batch, seq_len, num_items)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_items.reshape(-1))
            total_loss += loss.item()

            batch_size, seq_len, num_items = logits.shape
            # Gather the logits corresponding to the ground truth items
            target_scores = logits.gather(dim=-1, index=target_items.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len)
            # Calculate rank: number of items with a higher score than the target + 1
            ranks = (logits > target_scores.unsqueeze(-1)).sum(dim=-1) + 1  # (batch, seq_len)
            reciprocal_ranks = 1.0 / ranks.float()  # (batch, seq_len)
            # For recall, check if the ground truth is within the top_k predictions
            recall_hits = (ranks <= top_k).float()

            total_mrr += reciprocal_ranks.sum().item()
            total_recall += recall_hits.sum().item()
            total_count += batch_size * seq_len

    avg_loss = total_loss / len(dataloader)
    perplexity = np.exp(avg_loss)
    mrr = total_mrr / total_count
    map_score = mrr  # In this single-relevant-item setting, MAP is equivalent to MRR
    recall = total_recall / total_count

    print(f"Evaluation Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
    print(f"MRR: {mrr:.4f}, MAP: {map_score:.4f}, Recall@{top_k}: {recall:.4f}")
    return avg_loss, perplexity, mrr, map_score, recall

# ---------------------------
# Main Function: Putting It All Together
# ---------------------------
def main():
    data_path = "UserBehavior.csv"  # CSV file with columns: user_id, item_id, category_id, interaction_type, timestamp
    mode = "mixed"  # Options: "control", "sliding", "mixed"
    window_size = 100
    max_history = 500
    sliding_stride = 1
    batch_size = 16
    num_epochs = 5
    control_epochs = 2
    lr = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"

    colnames = ["user_id", "item_id", "category_id", "interaction_type", "timestamp"]
    df = pd.read_csv(data_path, names=colnames, header=None)
    
    # **Step 1: Compute item interaction frequency**
    item_counts = df["item_id"].value_counts()
    frequent_items = item_counts[item_counts >= 50].index  # Items with enough interactions
    df = df[df["item_id"].isin(frequent_items)]
    # Use the number of unique items (after filtering) as the number of classes
    num_items = df["item_id"].nunique()
    print("Number of raw items:", num_items)

    if mode in ["control", "sliding"]:
        dataset = InteractionDataset(data_path, mode=mode, window_size=window_size,
                                     max_history=(max_history if mode=="sliding" else None),
                                     sliding_stride=sliding_stride)

        print("Number of samples in dataset:", len(dataset))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    elif mode == "mixed":
        control_dataset = InteractionDataset(data_path, mode="control", window_size=window_size)
        sliding_dataset = InteractionDataset(data_path, mode="sliding", window_size=window_size,
                                             max_history=max_history, sliding_stride=sliding_stride)
        control_dataloader = DataLoader(control_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
        sliding_dataloader = DataLoader(sliding_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    else:
        raise ValueError("Invalid mode. Choose one of: 'control', 'sliding', 'mixed'.")
    
    print("Dataset loaded.")

    # The model's output dimension should match the number of unique (mapped) items.
    model = Gemma2(num_items=num_items, num_interaction_types=len(INTERACTION_TYPE_MAPPING),
                   emb_dim=32, n_layers=2, n_heads=4, dropout=0.1, max_seq_len=window_size)
    
    # model.load_state_dict(torch.load("taobaosliding1000.pth", map_location=device))
    # model.to(device)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters:", count_parameters(model))

    if mode == "mixed":
        train_model_mixed(control_dataloader, sliding_dataloader, model,
                          num_epochs=num_epochs, X=control_epochs, lr=lr, device=device)
    else:
        train_model(model, dataloader, num_epochs=num_epochs, lr=lr, device=device)
        
    print("Evaluating on sliding dataset:")
    if mode == "mixed":
        evaluate_model(model, sliding_dataloader, device=device)
    else:
        evaluate_model(model, dataloader, device=device)

if __name__ == "__main__":
    main()
