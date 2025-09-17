import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.special import expit  # sigmoid function

class NormalDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        feature_cols = [col for col in df.columns if col not in ['subject_id', 'death']]
        grouped = df.groupby('subject_id')

        self.inputs = []
        self.labels = []

        for subject_id, group in grouped:
            group = group.sort_index()  
            features = group[feature_cols].drop(columns=['death'], errors='ignore').values.astype(np.float32)
            label = group['death'].values[0]
            self.inputs.append(torch.tensor(features, dtype=torch.float32))
            self.labels.append(torch.tensor(label, dtype=torch.float32))
        
        self.inputs = torch.stack(self.inputs)
        self.labels = torch.stack(self.labels)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# ======================= Models ============================
class MLP(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape[0] * input_shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        _, (h_n, _) = self.rnn(x)
        return self.fc(h_n[-1])

class SAINT(nn.Module):
    def __init__(self, input_dim, seq_len, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.flatten(1)
        return self.fc(x)
    
    
def train_model(model, train_loader, test_loader, device, epochs=10):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    
    train_loss, test_loss = [], []

    best_epoch = 0
    best_test_loss = float('inf')
    best_preds = None
    best_labels = None
    best_state_dict = None

    for epoch in range(epochs):
        model.train()
        total = 0
        running_loss = 0.0
        for X, y in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}"):
            X, y = X.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
            total += X.size(0)
        train_loss.append(running_loss / total)

        model.eval()
        running_loss = 0.0
        preds_epoch = []
        labels_epoch = []
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device).unsqueeze(1)
                pred = model(X)
                loss = criterion(pred, y)
                running_loss += loss.item() * X.size(0)
                preds_epoch.extend(pred.cpu().numpy())
                labels_epoch.extend(y.cpu().numpy())
        epoch_test_loss = running_loss / len(test_loader.dataset)
        test_loss.append(epoch_test_loss)

        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            best_epoch = epoch
            best_preds = np.array(preds_epoch).flatten()
            best_labels = np.array(labels_epoch).flatten()
            best_state_dict = model.state_dict()

    model.load_state_dict(best_state_dict)
    return best_epoch, train_loss, test_loss, best_preds, best_labels


# ----------------- Evaluation Function -----------------
def evaluate(preds_logits, labels):
    preds_prob = expit(preds_logits)  # sigmoid activation
    preds_bin = (preds_prob >= 0.5).astype(int)
    return {
        "AUC": roc_auc_score(labels, preds_prob),
        "Accuracy": accuracy_score(labels, preds_bin),
        "F1": f1_score(labels, preds_bin),
        "Precision": precision_score(labels, preds_bin, zero_division=0),
        "Recall": recall_score(labels, preds_bin, zero_division=0)
    }

def main():
    csv_path = "data/preprocessed_data.csv"  
    dataset = NormalDataset(csv_path)
    input_shape = dataset[0][0].shape

    # 取得標籤，用於 stratify split
    labels = [int(dataset[i][1]) for i in range(len(dataset))]
    indices = list(range(len(dataset)))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=8
    )

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    models = {
        "MLP": MLP(input_shape),
        "RNN": RNNModel(input_size=input_shape[1]),
        "SAINT": SAINT(input_dim=input_shape[1], seq_len=input_shape[0])
    }

    history = {}
    metrics = {}
    best_epochs = {}
    for name, model in models.items():
        print(f"\n🚀 Training {name}")
        best_epoch, train_loss, test_loss, preds, labels = train_model(model, train_loader, test_loader, device, 40)
        result = evaluate(preds, labels)
        history[name] = (train_loss, test_loss)
        metrics[name] = result
        best_epochs[name] = best_epoch

    # Plot losses
    plt.figure(figsize=(10, 6))
    for name, (tr, te) in history.items():
        plt.plot(tr, label=f'{name} Train')
        plt.plot(te, label=f'{name} Test')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Test Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_comparison_without_timeline.png")
    print("📊 Saved: loss_comparison_without_timeline.png")

    # Print metrics
    df_metrics = pd.DataFrame(metrics).T
    print("\n📈 Evaluation Metrics:")
    print(df_metrics.round(4))
    df_metrics.to_csv("model_metrics_without_timeline.csv")
    print("📁 Saved: model_metrics.csv")
    print(best_epochs)

if __name__ == "__main__":
    main()
