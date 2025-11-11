import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('mnist_train.csv')
X = torch.tensor(df.iloc[:, 1:].values / 255.0, dtype=torch.float32)
y = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)

loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

# Build MLP with 2 hidden layers
model = nn.Sequential(
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 10)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training
epochs = 10
losses, accs = [], []

for epoch in range(epochs):
    total_loss, correct = 0, 0
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(xb)
        correct += (pred.argmax(1) == yb).sum().item()
    
    losses.append(total_loss / len(X))
    accs.append(correct / len(X))
    print(f"Epoch {epoch+1}: Loss={losses[-1]:.4f}, Acc={accs[-1]:.4f}")

# Plot curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(accs)
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
