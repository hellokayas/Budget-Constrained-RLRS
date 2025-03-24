import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# ---------------------------
# Dataset and Sampling Methods
# ---------------------------
class InteractionDataset(Dataset):
    """
    Dataset for user interaction sequences.
    
    Depending on the mode, this dataset generates training samples in one of three ways:
    
    - "control": For each user, use only the most recent window (last k interactions).
    - "sliding": For each user, generate sliding windows (of size k) over (optionally, a max portion of)
      the user’s entire history.
    - "mixed": (Not used directly here – the mixed approach is implemented by combining two datasets.)
    
    Each sample returns an input sequence (all tokens except the last) and the target sequence (the
    input sequence shifted by one token), which is typical for autoregressive next-item prediction.
    """
    def __init__(self, data_path, mode="control", window_size=100, max_history=None, sliding_stride=1):
        """
        Args:
          data_path: Path to a CSV file containing at least: user_id, item_id, timestamp.
          mode: One of "control" or "sliding".
          window_size: Length of the window (e.g. 100).
          max_history: For sliding mode, optionally limit to the last max_history interactions.
          sliding_stride: Stride when sliding the window.
        """
        self.df = pd.read_csv(data_path)
        self.df.sort_values(["user_id", "timestamp"], inplace=True)
        # Group interactions by user (each value is a list of item_ids in time order)
        self.user_sequences = self.df.groupby("user_id")["item_id"].apply(list).to_dict()
        self.mode = mode
        self.window_size = window_size
        self.max_history = max_history
        self.sliding_stride = sliding_stride

        # Build samples list: each sample is (user_id, window_sequence)
        self.samples = []
        for user, seq in self.user_sequences.items():
            if self.mode == "control":
                # Use only the most recent k items (or all if not enough)
                if len(seq) >= window_size:
                    sample = seq[-window_size:]
                else:
                    sample = seq
                self.samples.append((user, sample))
            elif self.mode == "sliding":
                # Optionally restrict the history (e.g., Mixed-500 or Mixed-1000)
                if self.max_history is not None:
                    seq = seq[-self.max_history:]
                # Generate sliding windows (if sequence length is less than window size, use entire seq)
                if len(seq) < window_size:
                    self.samples.append((user, seq))
                else:
                    for i in range(0, len(seq) - window_size + 1, self.sliding_stride):
                        window = seq[i:i + window_size]
                        self.samples.append((user, window))
            else:
                raise ValueError("Invalid mode. Use 'control' or 'sliding'.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        For autoregressive prediction we return:
          input_seq: all but the last token
          target_seq: sequence shifted left by one (i.e. next-token prediction)
        """
        user, window = self.samples[idx]
        window = torch.tensor(window, dtype=torch.long)
        # If window length is 1 (or less than 2) we cannot create input-target pair;
        # in practice, you may want to filter out such cases.
        if len(window) < 2:
            # Padding example (this simple example pads with 0; adjust as needed)
            window = torch.cat([window, torch.tensor([0], dtype=torch.long)])
        input_seq = window[:-1]
        target_seq = window[1:]
        return input_seq, target_seq

# ---------------------------
# Model Definition: Gemma2
# ---------------------------
class Gemma2(nn.Module):
    """
    A simple transformer-based model for next-item prediction.
    
    This model includes an item embedding, positional embedding, a transformer encoder
    (with causal masking to enforce autoregressive prediction), and a final linear layer that outputs logits over items.
    """
    def __init__(self, num_items, emb_dim=64, n_layers=2, n_heads=4, dropout=0.1, max_seq_len=100):
        super(Gemma2, self).__init__()
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        self.position_embedding = nn.Embedding(max_seq_len, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_layer = nn.Linear(emb_dim, num_items)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len) representing a sequence of item indices.
        Returns logits of shape (batch_size, seq_len, num_items).
        """
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.item_embedding(x) + self.position_embedding(positions)
        # Transformer expects (seq_len, batch_size, emb_dim)
        x = x.transpose(0, 1)
        # Create causal mask: each position cannot attend to future positions.
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        x = self.transformer(x, mask=mask)
        x = x.transpose(0, 1)  # Back to (batch_size, seq_len, emb_dim)
        logits = self.output_layer(x)
        return logits

# ---------------------------
# Training Functions
# ---------------------------
def train_model(model, dataloader, num_epochs=10, lr=0.001, device='cpu'):
    """
    Standard training loop used for either "control" (fixed window) or "all_sliding" mode.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            optimizer.zero_grad()
            logits = model(input_seq)  # logits: (batch, seq_len, num_items)
            # Compute loss: flatten the sequences so that each predicted token is a sample
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss / len(dataloader):.4f}")

def train_model_mixed(control_dataloader, sliding_dataloader, model, num_epochs=10, X=5, lr=0.001, device='cpu'):
    """
    Mixed training loop: for the first X epochs, we use control (fixed window) samples,
    and for the remaining epochs, we use sliding window samples.
    
    X should be chosen based on validation or hyperparameter sweep.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(num_epochs):
        if epoch < X:
            dataloader = control_dataloader
            print(f"Epoch {epoch + 1}/{num_epochs} using CONTROL (fixed window) sampling")
        else:
            dataloader = sliding_dataloader
            print(f"Epoch {epoch + 1}/{num_epochs} using SLIDING window sampling")
        
        epoch_loss = 0.0
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            optimizer.zero_grad()
            logits = model(input_seq)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(dataloader):.4f}")

def evaluate_model(model, dataloader, device='cpu'):
    """
    Evaluates the model by computing the average loss and perplexity on a validation set.
    (Note: Computing ranking metrics like MRR, mAP, and recall would require additional code
    to rank predictions and compare to held-out items.)
    """
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            logits = model(input_seq)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    perplexity = np.exp(avg_loss)
    print(f"Evaluation Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
    return avg_loss, perplexity

# ---------------------------
# Main Function: Putting It All Together
# ---------------------------
def main():
    # ----- Configuration -----
    data_path = "interactions.csv"  # CSV file with columns: user_id, item_id, timestamp
    mode = "mixed"  # Options: "control", "sliding", "mixed"
    window_size = 100
    # For mixed/sliding mode, you can choose a max_history limit.
    # For example, for Mixed-500 set max_history = 500, for Mixed-1000 set max_history = 1000.
    max_history = 500  
    sliding_stride = 1
    batch_size = 64
    num_epochs = 10
    # For mixed mode, use X epochs for control (fixed window) sampling
    control_epochs = 5  
    lr = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- Data Preparation -----
    # To determine the vocabulary size (number of items), we assume that item_id values are integers.
    df = pd.read_csv(data_path)
    num_items = int(df["item_id"].max()) + 1  # assuming item_ids start at 0

    if mode == "control" or mode == "sliding":
        dataset = InteractionDataset(data_path, mode=mode, window_size=window_size,
                                     max_history=(max_history if mode=="sliding" else None),
                                     sliding_stride=sliding_stride)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    elif mode == "mixed":
        # Create two datasets: one for control and one for sliding
        control_dataset = InteractionDataset(data_path, mode="control", window_size=window_size)
        sliding_dataset = InteractionDataset(data_path, mode="sliding", window_size=window_size,
                                             max_history=max_history, sliding_stride=sliding_stride)
        control_dataloader = DataLoader(control_dataset, batch_size=batch_size, shuffle=True)
        sliding_dataloader = DataLoader(sliding_dataset, batch_size=batch_size, shuffle=True)
    else:
        raise ValueError("Invalid mode. Choose one of: 'control', 'sliding', 'mixed'.")

    # ----- Model Initialization -----
    model = Gemma2(num_items=num_items, emb_dim=64, n_layers=2, n_heads=4, dropout=0.1, max_seq_len=window_size)

    # ----- Training -----
    if mode == "mixed":
        train_model_mixed(control_dataloader, sliding_dataloader, model,
                          num_epochs=num_epochs, X=control_epochs, lr=lr, device=device)
    else:
        train_model(model, dataloader, num_epochs=num_epochs, lr=lr, device=device)

    # ----- Evaluation (on the same data for illustration; ideally use a held-out set) -----
    if mode == "mixed":
        print("Evaluating on sliding dataset:")
        evaluate_model(model, sliding_dataloader, device=device)
    else:
        evaluate_model(model, dataloader, device=device)

    # Save the trained model
    torch.save(model.state_dict(), "gemma2_model.pth")
    print("Model saved as gemma2_model.pth")

if __name__ == "__main__":
    main()

"""
The current implementation is simplified and only expects a CSV with at least three columns: user_id, item_id, and timestamp. It doesn't explicitly require or use details about the interaction type (e.g., video plays, likes, add to watchlist) as long as these are encoded into the item_id or handled upstream. 

If your dataset contains more detailed features (like the type of interaction), you'll need to adjust the preprocessing or model to incorporate these extra signals—for example, by adding additional feature embeddings or by filtering/splitting data based on interaction type. The modified code for that new approach is given below.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# ---------------------------
# Interaction Type Mapping
# ---------------------------
# Define a mapping for the interaction types.
INTERACTION_TYPE_MAPPING = {
    "video plays": 0,
    "video likes": 1,
    "add to watchlist": 2,
    "open video details page": 3
}

# ---------------------------
# Dataset and Sampling Methods
# ---------------------------
class InteractionDataset(Dataset):
    """
    Dataset for user interaction sequences with additional features.
    
    Each interaction includes:
      - item_id
      - interaction_type (e.g. video plays, video likes, etc.)
    
    Depending on the mode, this dataset generates training samples as either:
    
    - "control": Use only the most recent window (last k interactions) for each user.
    - "sliding": Generate multiple sliding windows (of size k) over a user’s interaction history.
    
    Each sample returns:
      - input_items: Tensor of item ids (all but the last token)
      - target_items: Tensor of item ids shifted by one (for next-item prediction)
      - input_types: Tensor of interaction type ids corresponding to the input_items
    """
    def __init__(self, data_path, mode="control", window_size=100, max_history=None, sliding_stride=1):
        """
        Args:
          data_path: Path to a CSV file with columns: user_id, item_id, timestamp, interaction_type.
          mode: "control" or "sliding".
          window_size: Window length (e.g., 100).
          max_history: In sliding mode, optionally restrict to the last max_history interactions.
          sliding_stride: Step size for sliding window.
        """
        self.df = pd.read_csv(data_path)
        self.df.sort_values(["user_id", "timestamp"], inplace=True)
        
        # Build sequences per user as (items, interaction_types)
        self.user_sequences = {}
        for user, group in self.df.groupby("user_id"):
            group = group.sort_values("timestamp")
            items = group["item_id"].tolist()
            # Map interaction_type string to an integer using our mapping
            types = group["interaction_type"].map(lambda x: INTERACTION_TYPE_MAPPING.get(x, 0)).tolist()
            self.user_sequences[user] = (items, types)
        
        self.mode = mode
        self.window_size = window_size
        self.max_history = max_history
        self.sliding_stride = sliding_stride

        # Build list of samples: each sample is (user, item_seq, type_seq)
        self.samples = []
        for user, (items, types) in self.user_sequences.items():
            if self.mode == "control":
                # Use only the most recent window (last k interactions)
                if len(items) >= window_size:
                    sample_items = items[-window_size:]
                    sample_types = types[-window_size:]
                else:
                    sample_items = items
                    sample_types = types
                self.samples.append((user, sample_items, sample_types))
            elif self.mode == "sliding":
                # Optionally restrict to the last max_history interactions
                if self.max_history is not None:
                    items = items[-self.max_history:]
                    types = types[-self.max_history:]
                # Generate sliding windows
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
        """
        Returns:
          input_items: Tensor of shape (seq_len - 1)
          target_items: Tensor of shape (seq_len - 1)
          input_types: Tensor of shape (seq_len - 1)
        """
        user, item_seq, type_seq = self.samples[idx]
        items = torch.tensor(item_seq, dtype=torch.long)
        types = torch.tensor(type_seq, dtype=torch.long)
        # Ensure the sequence is at least length 2 for input-target pairing.
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
    """
    Transformer-based recommendation model that includes:
      - Item embedding
      - Interaction type embedding
      - Positional embedding
      
    The input representation is the sum of these embeddings. A transformer encoder with causal masking is used,
    and a final linear layer outputs logits over items for next-item prediction.
    """
    def __init__(self, num_items, num_interaction_types=4, emb_dim=64, n_layers=2, n_heads=4, dropout=0.1, max_seq_len=100):
        super(Gemma2, self).__init__()
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        self.interaction_embedding = nn.Embedding(num_interaction_types, emb_dim)
        self.position_embedding = nn.Embedding(max_seq_len, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_layer = nn.Linear(emb_dim, num_items)
        self.max_seq_len = max_seq_len

    def forward(self, x, interaction_types):
        """
        Args:
          x: Tensor of shape (batch_size, seq_len) with item ids.
          interaction_types: Tensor of shape (batch_size, seq_len) with interaction type ids.
          
        Returns:
          logits: Tensor of shape (batch_size, seq_len, num_items)
        """
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        item_emb = self.item_embedding(x)
        type_emb = self.interaction_embedding(interaction_types)
        pos_emb = self.position_embedding(positions)
        x = item_emb + type_emb + pos_emb
        # Transformer expects (seq_len, batch_size, emb_dim)
        x = x.transpose(0, 1)
        # Create causal mask so tokens cannot attend to future tokens
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        x = self.transformer(x, mask=mask)
        x = x.transpose(0, 1)  # Back to (batch_size, seq_len, emb_dim)
        logits = self.output_layer(x)
        return logits

# ---------------------------
# Training and Evaluation Functions
# ---------------------------
def train_model(model, dataloader, num_epochs=10, lr=0.001, device='cpu'):
    """
    Standard training loop for "control" or "sliding" modes.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for input_items, target_items, input_types in dataloader:
            input_items = input_items.to(device)
            target_items = target_items.to(device)
            input_types = input_types.to(device)
            optimizer.zero_grad()
            logits = model(input_items, input_types)  # logits: (batch, seq_len, num_items)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_items.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss / len(dataloader):.4f}")

def train_model_mixed(control_dataloader, sliding_dataloader, model, num_epochs=10, X=5, lr=0.001, device='cpu'):
    """
    Mixed training loop: the first X epochs use control (fixed window) samples,
    and subsequent epochs use sliding window samples.
    """
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

def evaluate_model(model, dataloader, device='cpu'):
    """
    Evaluate the model by computing the average loss and perplexity.
    """
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for input_items, target_items, input_types in dataloader:
            input_items = input_items.to(device)
            target_items = target_items.to(device)
            input_types = input_types.to(device)
            logits = model(input_items, input_types)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_items.reshape(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    perplexity = np.exp(avg_loss)
    print(f"Evaluation Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
    return avg_loss, perplexity

# ---------------------------
# Main Function: Putting It All Together
# ---------------------------
def main():
    # ----- Configuration -----
    data_path = "interactions_with_features.csv"  # CSV file with columns: user_id, item_id, timestamp, interaction_type
    mode = "mixed"  # Options: "control", "sliding", "mixed"
    window_size = 100
    max_history = 500  # e.g., for Mixed-500 setting
    sliding_stride = 1
    batch_size = 64
    num_epochs = 10
    control_epochs = 5  # Number of epochs using control sampling before switching in mixed mode
    lr = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- Data Preparation -----
    df = pd.read_csv(data_path)
    # Determine number of items (assuming item_ids are integers starting at 0)
    num_items = int(df["item_id"].max()) + 1

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

    # ----- Model Initialization -----
    model = Gemma2(num_items=num_items, num_interaction_types=len(INTERACTION_TYPE_MAPPING),
                   emb_dim=64, n_layers=2, n_heads=4, dropout=0.1, max_seq_len=window_size)

    # ----- Training -----
    if mode == "mixed":
        train_model_mixed(control_dataloader, sliding_dataloader, model,
                          num_epochs=num_epochs, X=control_epochs, lr=lr, device=device)
    else:
        train_model(model, dataloader, num_epochs=num_epochs, lr=lr, device=device)

    # ----- Evaluation -----
    if mode == "mixed":
        print("Evaluating on sliding dataset:")
        evaluate_model(model, sliding_dataloader, device=device)
    else:
        evaluate_model(model, dataloader, device=device)

    # Save the trained model
    torch.save(model.state_dict(), "gemma2_model_with_features.pth")
    print("Model saved as gemma2_model_with_features.pth")

if __name__ == "__main__":
    main()
