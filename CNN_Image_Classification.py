import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('mnist_train.csv')
X = torch.tensor(df.iloc[:3000, 1:].values.reshape(-1, 1, 28, 28) / 255.0, dtype=torch.float32)
y = torch.tensor(df.iloc[:3000, 0].values, dtype=torch.long)
loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

# CNN (2 conv + FC)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc = nn.Linear(64*5*5, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = torch.relu(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        return self.fc(x.view(-1, 64*5*5))

# MLP for comparison
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x.view(-1, 784))))

# Data augmentation
def augment(x):
    if torch.rand(1) > 0.5: x = torch.flip(x, [3])  # flip
    return x

def train(model, use_aug=False):
    opt = torch.optim.Adam(model.parameters())
    accs = []
    for epoch in range(5):
        correct = 0
        for xb, yb in loader:
            if use_aug: xb = augment(xb)
            pred = model(xb)
            loss = nn.CrossEntropyLoss()(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (pred.argmax(1) == yb).sum().item()
        accs.append(correct / len(X))
    return accs

# Train both models
cnn_acc = train(CNN(), use_aug=True)
mlp_acc = train(MLP())

# Plot comparison
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(cnn_acc, label='CNN (with augmentation)')
plt.plot(mlp_acc, label='MLP')
plt.title('Accuracy Comparison')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.bar(['CNN', 'MLP'], [cnn_acc[-1], mlp_acc[-1]])
plt.title('Final Accuracy')
plt.ylim(0, 1)
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"CNN: {cnn_acc[-1]:.3f} | MLP: {mlp_acc[-1]:.3f}")
