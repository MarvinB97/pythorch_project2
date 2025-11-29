import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# config
device = torch.device("cpu")  # SOLO CPU

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# cargar dataset
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# modelo mlp
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),     # 28x28 → 784
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 números
        )

    def forward(self, x):
        return self.layers(x)

model = MLP().to(device)

# training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 3

print("Entrenando...\n")
for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Época {epoch+1}/{epochs} - Pérdida: {total_loss:.4f}")

torch.save(model.state_dict(), "model.pth")
print("\nModelo entrenado, se guardó como model.pth\n")

# evaluacion
correct = 0
total = 0

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total * 100
print(f"Precisión en test: {accuracy:.2f}%\n")

# test manuales
def probar_manual(indice):
    """
    Muestra una imagen del dataset de test y su predicción.
    """
    img, label = test_data[indice]
    model.eval()
    with torch.no_grad():
        salida = model(img.unsqueeze(0))
        pred = torch.argmax(salida).item()

    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"Etiqueta real: {label} - Predicción: {pred}")
    plt.show()

# Ejemplo: probar el índice 0
probar_manual(0)
