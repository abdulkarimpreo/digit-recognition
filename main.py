import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pathlib import Path
from src.data_loader import load_data
from src.model import create_model
from src.train import train_model
from src.evaluate import evaluate_model

# Load Data
train_loader, test_loader, test_data = load_data()


if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name(0))
else:
    print("CPU")


device ="cuda" if torch.cuda.is_available() else "cpu"
print("Using device", device)

# Model
model = create_model().to(device)

# optim and Loss
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= 0.001)

# Model Train 
train_model(model, optimizer, loss_fn, train_loader, device)

# Calculate accuracy
accuracy = evaluate_model(model, test_loader, device)
print(f"Total Accuracy: {accuracy:.2f}%")


# Save the trained model
BASE_DIR = Path(__file__).resolve().parent
model_file = BASE_DIR / "models" / "mnist_model.pth"
torch.save(model.state_dict(), model_file)

print("Model saved successfully!")

# Predict
index = 4
image, true_label = test_data[index]

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Actual Lable : {true_label}")
plt.axis("off")

image_flat = image.view(1, -1).to(device)

with torch.no_grad():
    output = model(image_flat)
    predicted_label = output.argmax(dim=1).item()

print("User picked image index", index)
print("Actual Label", true_label)
print("Model Prediction", predicted_label)
plt.show()