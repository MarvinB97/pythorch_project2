import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- CONFIG ----------------
device = torch.device("cpu")  # Solo CPU

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ---------------- CARGA SEGURA DE IMÁGENES ----------------
def pil_loader_safe(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except:
        print(f"Archivo corrupto ignorado: {path}")
        return None

class ImageFolderSafe(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader_safe(path)
        if sample is None:
            # Si la imagen está corrupta, tomar la siguiente
            return self.__getitem__((index + 1) % len(self.samples))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

# Dataset
dataset = ImageFolderSafe("data", transform=transform)
print("Clases:", dataset.classes)

# ---------------- DIVISIÓN TRAIN/TEST ----------------
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---------------- VISUALIZAR IMAGEN ----------------
images, labels = next(iter(train_loader))
img = images[0].permute(1, 2, 0) * 0.5 + 0.5  # desnormalizar
plt.imshow(img)
plt.title(f"Label: {dataset.classes[labels[0]]}")
plt.axis("off")
plt.show()

# ---------------- MODELO CNN ----------------
model = nn.Sequential(
    # Bloque 1
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    # Bloque 2
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    # Bloque 3
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    # Aplanar
    nn.Flatten(),

    # Clasificación
    nn.Linear(128 * 28 * 28, 256),
    nn.ReLU(),
    nn.Linear(256, len(dataset.classes))  # Número de clases
)

model.to(device)

# ---------------- ENTRENAMIENTO ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 3

print("----------- Entrenando -----------\n")
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
print("\nModelo entrenado y guardado como model.pth\n")

# ---------------- EVALUACIÓN ----------------
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total * 100
print(f"Precisión en test: {accuracy:.2f}%\n")