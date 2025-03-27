
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

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
        colnames = ["visitorid","itemid","event","timestamp","transactionid"]
        self.df = pd.read_csv(data_path,nrows=1000)
        print(self.df.head())
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
        if len(items) < 2:
            items = torch.cat([items, torch.tensor([0], dtype=torch.long)])
            types = torch.cat([types, torch.tensor([0], dtype=torch.long)])
        input_items = items[:-1]
        target_items = items[1:]
        input_types = types[:-1]
        return input_items, target_items, input_types

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
        # print(item_emb.shape,type_emb.shape,pos_emb.shape)
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
        for input_items, target_items, input_types in dataloader:
            # print(input_items,target_items,input_types)
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
        for input_items, target_items, input_types in dataloader:
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
        for input_items, target_items, input_types in dataloader:
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
    data_path = "events.csv"  # CSV file with columns: visitorid, itemid, timestamp, event
    mode = "control"  # Options: "control", "sliding", "mixed"
    window_size = 100
    max_history = 500
    sliding_stride = 1
    batch_size = 1
    num_epochs = 1
    control_epochs = 5
    lr = 0.001
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    colnames = ["visitorid","itemid","category_id","event","timestamp"]
    df = pd.read_csv(data_path,nrows=1000)
    print(df.head())
    num_items = df["itemid"].max() + 1

    if mode in ["control", "sliding"]:
        dataset = InteractionDataset(data_path, mode=mode, window_size=window_size,
                                     max_history=(max_history if mode=="sliding" else None),
                                     sliding_stride=sliding_stride)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    elif mode == "mixed":
        control_dataset = InteractionDataset(data_path, mode="control", window_size=window_size)
        sliding_dataset = InteractionDataset(data_path, mode="sliding", window_size=window_size,
                                             max_history=max_history, sliding_stride=sliding_stride)
        control_dataloader = DataLoader(control_dataset, batch_size=batch_size, shuffle=True)
        sliding_dataloader = DataLoader(sliding_dataset, batch_size=batch_size, shuffle=True)
    else:
        raise ValueError("Invalid mode. Choose one of: 'control', 'sliding', 'mixed'.")

    model = Gemma2(num_items=num_items, num_events=len(event_MAPPING),
                   emb_dim=32, n_layers=2, n_heads=4, dropout=0.1, max_seq_len=window_size)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(count_parameters(model))

    if mode == "mixed":
        train_model_mixed(control_dataloader, sliding_dataloader, model,
                          num_epochs=num_epochs, X=control_epochs, lr=lr, device=device)
    else:
        train_model(model, dataloader, num_epochs=num_epochs, lr=lr, device=device)

    if mode == "mixed":
        print("Evaluating on sliding dataset:")
        evaluate_model(model, sliding_dataloader, device=device)
    else:
        evaluate_model(model, dataloader, device=device)

    torch.save(model.state_dict(), "gemma2_model_with_features.pth")
    print("Model saved as gemma2_model_with_features.pth")

if __name__ == "__main__":
    main()
