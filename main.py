import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 🧠 Başlangıç parametreleri
input_size = 28 * 28
hidden_size = 64
output_size = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GrowLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(GrowLinear, self).__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features, device=device) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features, device=device))

    def forward(self, x):
        return F.linear(x, self.weights, self.bias)

    def grow(self, new_neurons):
        old_out, in_features = self.weights.shape
        new_weights = torch.randn(new_neurons, in_features, device=self.weights.device) * 0.01
        new_bias = torch.zeros(new_neurons, device=self.bias.device)

        self.weights = nn.Parameter(torch.cat([self.weights, new_weights], dim=0))
        self.bias = nn.Parameter(torch.cat([self.bias, new_bias], dim=0))

class GrowNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GrowNet, self).__init__()
        self.layer1 = GrowLinear(input_size, hidden_size)
        self.layer2 = GrowLinear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

    def grow_brain_cells(self, new_neurons):
        self.layer1.grow(new_neurons)

        # Yeni gelen nöronlara göre layer2'ye giriş ekle
        old_out, old_in = self.layer2.weights.shape
        new_inputs = torch.randn(old_out, new_neurons, device=self.layer2.weights.device) * 0.01
        self.layer2.weights = nn.Parameter(torch.cat([self.layer2.weights, new_inputs], dim=1))

# 🔄 Dataset
transform = transforms.ToTensor()
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

# Test verisi
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

# 🚀 Eğitim
model = GrowNet(input_size, hidden_size, output_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

print("Training...")
for epoch in range(5):
    total, correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 🧠 Doğruluk
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f"🎯 Epoch {epoch+1} Accuracy: {acc:.4f}")

    # Her epoch sonunda beyin büyüsün!
    grow_amount = 8
    model.grow_brain_cells(grow_amount)
    print(f"🧠 +{grow_amount} nöron eklendi! Yeni gizli katman boyutu: {model.layer1.weights.shape[0]}")

# 🧠 Test Aşaması
y_true, y_pred = [], []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Confusion Matrix Görselleştirme
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("🧠 Confusion Matrix")
plt.show()
