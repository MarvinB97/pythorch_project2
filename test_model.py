import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt

# Device
device = torch.device("cpu")

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar
    transforms.ToTensor(),          # Convertir a tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizar
])

# Loader seguro para evitar crasheos con imágenes corruptas
def pil_loader_safe(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB') # Convierte a RGB
    except:
        print(f"Archivo corrupto ignorado: {path}")
        return None

# Dataset personalizado que ignora imágenes corruptas
class ImageFolderSafe(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader_safe(path)
        if sample is None:
            return self.__getitem__((index + 1) % len(self.samples))
        if self.transform:
            sample = self.transform(sample)
        return sample, target

# Cargar dataset desde "data"
dataset = ImageFolderSafe("data", transform=transform)
print("Clases:", dataset.classes)

# Dividir en train/test 80/20
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Arquitectura CNN simple
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Flatten(),

    nn.Linear(128 * 28 * 28, 256),
    nn.ReLU(),
    nn.Linear(256, len(dataset.classes))    # salida depende del número de clases
)

# Mandar modelo al device
model.to(device)

# Cargar pesos entrenados
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

print("Modelo cargado correctamente.")


def probar_manual(indice):
    # Obtener imagen y etiqueta real
    img, label = test_dataset[indice]

    # Obtener predicción
    with torch.no_grad():
        salida = model(img.unsqueeze(0).to(device))
        pred = torch.argmax(salida).item()

    # Convertir tensor a imagen visible
    img_vis = img.permute(1, 2, 0) * 0.5 + 0.5

    # Mostrar
    plt.imshow(img_vis)
    plt.title(f"Real: {dataset.classes[label]} - Predicción: {dataset.classes[pred]}")
    plt.axis("off")
    plt.show()

# Probar índice 10
probar_manual(10)