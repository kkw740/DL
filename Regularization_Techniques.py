import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('mnist_train.csv')
X = torch.tensor(df.iloc[:3000, 1:].values / 255.0, dtype=torch.float32)
y = torch.tensor(df.iloc[:3000, 0].values, dtype=torch.long)

split = int(0.8 * len(X))
train_loader = DataLoader(TensorDataset(X[:split], y[:split]), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X[split:], y[split:]), batch_size=64)

# Simple MLP
class Model(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.drop = nn.Dropout(0.5) if dropout else None
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        if self.drop: x = self.drop(x)
        return self.fc2(x)

def train(model, l2=0):
    opt = torch.optim.Adam(model.parameters(), weight_decay=l2)
    loss_fn = nn.CrossEntropyLoss()
    train_loss, val_loss, val_acc = [], [], []
    
    for epoch in range(10):
        model.train()
        for xb, yb in train_loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        model.eval()
        tloss, vloss, correct = 0, 0, 0
        with torch.no_grad():
            for xb, yb in train_loader:
                tloss += loss_fn(model(xb), yb).item()
            for xb, yb in val_loader:
                pred = model(xb)
                vloss += loss_fn(pred, yb).item()
                correct += (pred.argmax(1) == yb).sum().item()
        
        train_loss.append(tloss / len(train_loader))
        val_loss.append(vloss / len(val_loader))
        val_acc.append(correct / len(X[split:]))
    
    return train_loss, val_loss, val_acc

# Train 3 models
print("Training: No Regularization")
no_reg = train(Model())

print("Training: L2 Regularization")
l2_reg = train(Model(), l2=0.01)

print("Training: Dropout")
dropout_reg = train(Model(dropout=True))

results = [("No Reg", no_reg), ("L2", l2_reg), ("Dropout", dropout_reg)]

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for name, (tl, vl, va) in results:
    axes[0].plot(tl, label=name)
    axes[1].plot(vl, label=name)
    axes[2].plot(va, label=name)

axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].set_title('Validation Loss')
axes[1].legend()
axes[1].grid(True)

axes[2].set_title('Validation Accuracy')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()
