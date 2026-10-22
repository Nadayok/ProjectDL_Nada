import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from model import CustomFashionCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


model = CustomFashionCNN().to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

all_preds, all_labels = [], []
failures = []

with torch.no_grad():
    for images, labels in test_loader:
        images_dev, labels_dev = images.to(device), labels.to(device)
        outputs = model(images_dev)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        for img, label, pred in zip(images, labels, preds):
            if label != pred and len(failures) < 5:
                failures.append((img, label.item(), pred.item()))

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))