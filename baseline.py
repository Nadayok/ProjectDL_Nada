import torch
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression

# Load data for baseline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

X_train = train_dataset.data[:50000].reshape(50000, -1).numpy() / 255.0
y_train = train_dataset.targets[:50000].numpy()
X_test = test_dataset.data.reshape(10000, -1).numpy() / 255.0
y_test = test_dataset.targets.numpy()

# Train and evaluate baseline
baseline = LogisticRegression(max_iter=100)
baseline.fit(X_train, y_train)
baseline_acc = baseline.score(X_test, y_test)

print(f"Baseline (Logistic Regression) Test Accuracy: {baseline_acc * 100:.2f}%")