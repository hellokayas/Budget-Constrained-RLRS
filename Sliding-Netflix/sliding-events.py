

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

# ---------------------------
# Interaction Type Mapping
# ---------------------------
event_MAPPING = {
    "view": 0,
    "addtocart": 1,
    "transaction": 2
}

# ---------------------------
# Dataset and Sampling Methods
# ---------------------------
class InteractionDataset(Dataset):
    def __init__(self, data_path, mode="control", window_size=100, max_history=None, sliding_stride=1):
        colnames = ["visitorid", "itemid", "event", "timestamp", "transactionid"]
        self.df = pd.read_csv(data_path)
        self.df.sort_values(["visitorid", "timestamp"], inplace=True)
        
        # Build sequences per user as (items, events)
        self.user_sequences = {}
        for user, group in self.df.groupby("visitorid"):
            group = group.sort_values("timestamp")
            items = group["itemid"].tolist()
            types = group["event"].map(lambda x: event_MAPPING.get(x, 0)).tolist()
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
        items = torch.tensor(item_seq, dtype=torch.long)
        types = torch.tensor(type_seq, dtype=torch.long)
        # Ensure that there is at least 2 elements to create input/target pairs
        if len(items) < 2:
            items = torch.cat([items, torch.tensor([0], dtype=torch.long)])
            types = torch.cat([types, torch.tensor([0], dtype=torch.long)])
        input_items = items[:-1]
        target_items = items[1:]
        input_types = types[:-1]
        return input_items, target_items, input_types

# ---------------------------
# Custom Collate Function for Padding
# ---------------------------
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
# Model Definition: Gemma2 with Interaction Feature Embedding
# ---------------------------
class Gemma2(nn.Module):
    def __init__(self, num_items, num_events=3, emb_dim=64, n_layers=2, n_heads=4, dropout=0.1, max_seq_len=100):
        super(Gemma2, self).__init__()
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        self.interaction_embedding = nn.Embedding(num_events, emb_dim)
        self.position_embedding = nn.Embedding(max_seq_len, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_layer = nn.Linear(emb_dim, num_items)
        self.max_seq_len = max_seq_len

    def forward(self, x, events):
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        item_emb = self.item_embedding(x)
        type_emb = self.interaction_embedding(events)
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
            # print(input_items.shape)
            optimizer.zero_grad()
            logits = model(input_items, input_types)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_items.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), "sliding500.pth")
        print("Model saved as sliding500.pth")

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
        torch.save(model.state_dict(), "1000_mixed.pth")
        print("Model saved as 1000_mixed.pth")

def evaluate_model(model, dataloader, device='cpu', top_k=10):
    """
    Evaluate the model by computing the average loss, perplexity, MRR, MAP, and Recall@top_k.
    This version computes MAP as the mean of per-query Average Precision (AP).
    If each query has only one ground-truth item, AP equals the reciprocal rank.
    """
    model.eval()
    total_loss = 0.0
    total_mrr = 0.0
    total_ap = 0.0
    total_queries = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    # We'll accumulate loss and ranking metrics over each query (each time step)
    with torch.no_grad():
        for input_items, target_items, input_types in dataloader:
            input_items = input_items.to(device)
            target_items = target_items.to(device)
            input_types = input_types.to(device)
            logits = model(input_items, input_types)  # Shape: (batch, seq_len, num_items)
            loss = criterion(logits.view(-1, logits.size(-1)), target_items.view(-1))
            total_loss += loss.item()
            
            batch_size, seq_len, num_items = logits.shape
            # Process each prediction (each time step) as a separate query
            for b in range(batch_size):
                for t in range(seq_len):
                    # Get the scores for all items for this query
                    scores = logits[b, t, :].cpu().numpy()
                    # Here we assume target_items[b, t] is the ground truth.
                    # If you have multiple relevant items, ensure that target_items[b, t]
                    # is a list or array of relevant item ids. If it's a single item, we wrap it.
                    gt = target_items[b, t].item()
                    ground_truth = [gt] if not isinstance(gt, (list, np.ndarray)) else gt
                    
                    # Sort all items by descending score
                    sorted_indices = np.argsort(-scores)
                    
                    # --- Compute Reciprocal Rank (for MRR) ---
                    # Find the rank of the first relevant item.
                    rank = np.where(np.isin(sorted_indices, ground_truth))[0][0] + 1
                    total_mrr += 1.0 / rank
                    
                    # --- Compute Average Precision (AP) for this query ---
                    hit_count = 0
                    precision_accum = 0.0
                    for i, idx in enumerate(sorted_indices, start=1):
                        if idx in ground_truth:
                            hit_count += 1
                            precision_accum += hit_count / i
                    # If no relevant item was found, AP is defined as 0.
                    ap = precision_accum / hit_count if hit_count > 0 else 0.0
                    total_ap += ap
                    
                    total_queries += 1
    
    avg_loss = total_loss / len(dataloader)
    perplexity = np.exp(avg_loss)
    mrr = total_mrr / total_queries
    map_score = total_ap / total_queries
    
    # Compute Recall@top_k as before (per query)
    total_recall = 0.0
    total_count = 0
    with torch.no_grad():
        for input_items, target_items, input_types in dataloader:
            input_items = input_items.to(device)
            target_items = target_items.to(device)
            input_types = input_types.to(device)
            logits = model(input_items, input_types)
            batch_size, seq_len, num_items = logits.shape
            # For each query, check if a relevant item appears in the top_k predictions.
            target_scores = logits.gather(dim=-1, index=target_items.unsqueeze(-1)).squeeze(-1)
            ranks = (logits > target_scores.unsqueeze(-1)).sum(dim=-1) + 1
            recall_hits = (ranks <= top_k).float()
            total_recall += recall_hits.sum().item()
            total_count += batch_size * seq_len
    recall = total_recall / total_count

    print(f"Evaluation Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
    print(f"MRR: {mrr:.4f}, MAP: {map_score:.4f}, Recall@{top_k}: {recall:.4f}")
    
    return avg_loss, perplexity, mrr, map_score, recall


# ---------------------------
# Main Function: Putting It All Together
# ---------------------------
def main():
    data_path = "events.csv"  # CSV file with columns: visitorid, itemid, timestamp, event
    mode = "mixed"  # Options: "control", "sliding", "mixed"
    window_size = 100
    max_history = 500
    sliding_stride = 1
    batch_size = 32  # Updated batch size
    num_epochs = 10
    control_epochs = 5
    lr = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(data_path)
    num_items = df["itemid"].max() + 1
    print(num_items)

    if mode in ["control", "sliding"]:
        dataset = InteractionDataset(data_path, mode=mode, window_size=window_size,
                                     max_history=(max_history if mode=="sliding" else None),
                                     sliding_stride=sliding_stride)
        print(len(dataset))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    elif mode == "mixed":
        control_dataset = InteractionDataset(data_path, mode="control", window_size=window_size)
        sliding_dataset = InteractionDataset(data_path, mode="sliding", window_size=window_size,
                                             max_history=max_history, sliding_stride=sliding_stride)
        control_dataloader = DataLoader(control_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
        sliding_dataloader = DataLoader(sliding_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    else:
        raise ValueError("Invalid mode. Choose one of: 'control', 'sliding', 'mixed'.")
    
    print("Dataset loaded")

    model = Gemma2(num_items=num_items, num_events=len(event_MAPPING),
                   emb_dim=32, n_layers=2, n_heads=4, dropout=0.1, max_seq_len=window_size)
    model.load_state_dict(torch.load("500_mixed.pth", weights_only=True))
    model.to(device)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters:", count_parameters(model))

    # if mode == "mixed":
    #     train_model_mixed(control_dataloader, sliding_dataloader, model,
    #                       num_epochs=num_epochs, X=control_epochs, lr=lr, device=device)
    # else:
    #     train_model(model, dataloader, num_epochs=num_epochs, lr=lr, device=device)

    if mode == "mixed":
        print("Evaluating on sliding dataset:")
        evaluate_model(model, sliding_dataloader, device=device)
    else:
        evaluate_model(model, dataloader, device=device)

    

if __name__ == "__main__":
    main()

