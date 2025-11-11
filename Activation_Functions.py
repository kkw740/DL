import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time

# Load data
df = pd.read_csv('mnist_train.csv')
X = torch.tensor(df.iloc[:2000, 1:].values / 255.0, dtype=torch.float32)
y = torch.tensor(df.iloc[:2000, 0].values, dtype=torch.long)
loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

# Simple MLP
class MLP(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.act = act
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

def train(act_name, act_fn):
    model = MLP(act_fn)
    opt = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    accs, grads = [], []
    
    start = time.time()
    for epoch in range(5):
        correct = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            
            grads.append(sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None))
            opt.step()
            correct += (pred.argmax(1) == yb).sum().item()
        accs.append(correct / len(X))
    
    return accs, grads[:150], time.time() - start

# Train with 3 activations
acts = {'Sigmoid': nn.Sigmoid(), 'Tanh': nn.Tanh(), 'ReLU': nn.ReLU()}
results = {name: train(name, act) for name, act in acts.items()}

# Plot
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

for name, (acc, grad, t) in results.items():
    ax[0].plot(acc, label=name)
    ax[1].plot(grad, label=name, alpha=0.7)

ax[0].set_title('Accuracy')
ax[0].legend()
ax[0].grid(True)

ax[1].set_title('Gradient Flow')
ax[1].legend()
ax[1].grid(True)

ax[2].bar(results.keys(), [r[2] for r in results.values()])
ax[2].set_title('Training Speed (sec)')
ax[2].grid(True)

plt.tight_layout()
plt.show()

print("\nVanishing Gradient Discussion:")
print("Sigmoid: Gradients < 0.25 → Vanishing problem")
print("Tanh:    Gradients < 1.0 → Still vanishes")
print("ReLU:    Gradients constant → No vanishing")
