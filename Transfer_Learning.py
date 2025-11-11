import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Load data
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train_loader = DataLoader(ImageFolder('dataset/train', transform=transform), batch_size=16, shuffle=True)
val_loader = DataLoader(ImageFolder('dataset/val', transform=transform), batch_size=16)

# Load pretrained VGG16
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, 2)

def train(freeze, epochs=3):
    for p in model.parameters(): p.requires_grad = not freeze
    for p in model.classifier.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    accs = []
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            loss = nn.CrossEntropyLoss()(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        correct = sum((model(x).argmax(1) == y).sum() for x, y in val_loader)
        acc = correct.item() / len(val_loader.dataset)
        accs.append(acc)
        print(f"Epoch {epoch+1}: {acc:.3f}")
    return accs

# Experiment
print("Frozen layers:"); frozen = train(True)
print("Fine-tuned:"); finetuned = train(False)

# Visualize
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(frozen, 'o-', label='Frozen')
plt.plot(finetuned, 's-', label='Fine-tuned')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
plt.subplot(1, 2, 2)
plt.bar(['Frozen', 'Fine-tuned'], [frozen[-1], finetuned[-1]])
plt.ylabel('Accuracy'); plt.ylim(0, 1); plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Result: Frozen={frozen[-1]:.3f}, Fine-tuned={finetuned[-1]:.3f}")
