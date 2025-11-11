import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load IMDB dataset
df = pd.read_csv('IMDB Dataset.csv').sample(3000, random_state=42).reset_index(drop=True)

# Tokenize
vocab = {}
X = [[vocab.setdefault(w, len(vocab)) for w in text.lower().split()[:80]] for text in df['review']]
X = [seq + [0]*(80-len(seq)) for seq in X]  # Pad to 80
y = (df['sentiment'] == 'positive').astype(int).values

# Split
split = int(0.8 * len(X))
X_train = torch.tensor(X[:split], dtype=torch.long)
y_train = torch.tensor(y[:split], dtype=torch.float32)
X_test = torch.tensor(X[split:], dtype=torch.long)
y_test = torch.tensor(y[split:], dtype=torch.float32)

loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

# LSTM Model
class LSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 32)
        self.lstm = nn.LSTM(32, 16, batch_first=True)
        self.fc = nn.Linear(16, 1)
    
    def forward(self, x):
        _, (h, _) = self.lstm(self.embed(x))
        return torch.sigmoid(self.fc(h[-1]))

model = LSTM(len(vocab))
opt = torch.optim.Adam(model.parameters())

# Train
accs = []
for epoch in range(5):
    correct = 0
    for xb, yb in loader:
        pred = model(xb).squeeze()
        loss = nn.BCELoss()(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        correct += ((pred > 0.5) == yb).sum().item()
    accs.append(correct / len(X_train))
    print(f"Epoch {epoch+1}: Accuracy = {accs[-1]:.3f}")

# Visualize
model.eval()
preds = model(X_test[:5]).squeeze().detach()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(accs)
plt.title('LSTM Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.bar(range(5), preds, alpha=0.7, label='Predicted')
plt.scatter(range(5), y_test[:5], color='red', s=100, label='True')
plt.title('Example Predictions')
plt.ylabel('Sentiment (1=Pos, 0=Neg)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Show examples
print("\nSample Predictions:")
for i in range(3):
    review = df.iloc[split+i]['review'][:100]
    true_label = "Positive" if y_test[i] == 1 else "Negative"
    pred_score = preds[i].item()
    pred_label = "Positive" if pred_score > 0.5 else "Negative"
    print(f"\nReview {i+1}: {review}...")
    print(f"True: {true_label} | Predicted: {pred_score:.3f} ({pred_label})")
