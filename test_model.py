import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# usar el modelo ya entrenado

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.layers(x)

device = torch.device("cpu")
model = MLP().to(device)

# Cargar pesos
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

print("el Modelo se cargó correctamente.")

# usar mnist para pruebas

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_data = datasets.MNIST(root="./data", train=False, transform=transform)

# funcion para probar la imagen

def probar_manual(indice):
    img, label = test_data[indice]

    with torch.no_grad():
        salida = model(img.unsqueeze(0))
        pred = torch.argmax(salida).item()

    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"Real: {label} - Predicción: {pred}")
    plt.show()

# test de ejemplo
probar_manual(10)
