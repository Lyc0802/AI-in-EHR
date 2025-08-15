import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from scipy.special import expit  # sigmoid function

# ----------------- Dataset -----------------
class TimeSeriesDataset(Dataset):
    def __init__(self, tensor_dict, label_dict):
        self.inputs = []
        self.labels = []
        for key in tensor_dict:
            x = tensor_dict[key]
            y = label_dict[key][0]
            y_clean = 1.0 if y >= 0.5 else 0.0  # 保證是 0 或 1
            self.inputs.append(torch.tensor(x, dtype=torch.float32))
            self.labels.append(torch.tensor(y_clean, dtype=torch.float32))
        self.inputs = torch.stack(self.inputs)
        self.labels = torch.stack(self.labels)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# ----------------- Models -----------------
class MLP(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape[0] * input_shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # no Sigmoid
        )

    def forward(self, x):
        return self.net(x)

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # no Sigmoid
        )

    def forward(self, x):
        _, (h_n, _) = self.rnn(x)
        return self.net(h_n[-1])

class SAINT(nn.Module):
    def __init__(self, input_dim, seq_len, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # no Sigmoid
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.flatten(1)
        return self.net(x)

# ----------------- Training Function -----------------
def train_model(model, train_loader, test_loader, device, epochs=40):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    train_loss, test_loss = [], []
    best_preds, best_labels = [], []
    best_test_loss = float('inf')
    best_epoch = 0
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
            best_preds = preds_epoch
            best_labels = labels_epoch
            best_epoch = epoch

    return best_epoch, train_loss, test_loss, np.array(best_preds).flatten(), np.array(best_labels).flatten()


# ----------------- Evaluation Function -----------------
def evaluate(preds_logits, labels):
    preds_prob = expit(preds_logits)  # sigmoid activation
    preds_bin = (preds_prob >= 0.5).astype(int)
    return {
        "AUC": roc_auc_score(labels, preds_prob),
        "F1": f1_score(labels, preds_bin),
        "Precision": precision_score(labels, preds_bin),
        "Recall": recall_score(labels, preds_bin)
    }

# ----------------- Main -----------------
def main():
    # Load data
    npz = np.load("timeseries_input_label.npz", allow_pickle=True)
    tensor_dict = npz['inputs'].item()
    label_dict = npz['labels'].item()
    dataset = TimeSeriesDataset(tensor_dict, label_dict)
    input_shape = dataset[0][0].shape

    # Split & loader
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64)

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
        best_epoch , train_loss, test_loss, preds, labels = train_model(model, train_loader, test_loader, device)
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
    plt.savefig("loss_comparison.png")
    print("📊 Saved: loss_comparison.png")

    # Print metrics
    df_metrics = pd.DataFrame(metrics).T
    print("\n📈 Evaluation Metrics:")
    print(df_metrics.round(4))
    df_metrics.to_csv("model_metrics.csv")
    print("📁 Saved: model_metrics.csv")

    # Descriptive analysis
    print("\n📊 Descriptive analysis:")
    death_labels = [label_dict[k][0] for k in label_dict]
    genders = [label_dict[k][1] for k in label_dict]
    print(f"Total samples: {len(death_labels)}")
    print(f"Death rate: {np.mean(death_labels):.2%}")
    print(f"Gender distribution:\n{pd.Series(genders).value_counts(normalize=True)}")

    # Plot distributions
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.countplot(x=death_labels)
    plt.title("Death Label Distribution")
    plt.savefig("death_distribution.png")

    plt.figure(figsize=(6, 4))
    sns.countplot(x=genders)
    plt.title("Gender Distribution")
    plt.savefig("gender_distribution.png")

    print("✅ Saved: death_distribution.png, gender_distribution.png")
    print(best_epochs)

if __name__ == "__main__":
    main()