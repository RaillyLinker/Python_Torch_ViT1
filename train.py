import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from vit import ViTClassifier  # ViTClassifier는 vit.py에 정의되어 있다고 가정
from torch.utils.data import Subset

BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_dataset = Subset(train_dataset, range(100))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = ViTClassifier(
        img_size=IMG_SIZE,
        patch_size=16,
        in_channels=3,
        emb_size=256,
        n_layers=6,
        attn_heads=8,
        dropout=0.1,
        num_classes=10
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("train process start")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        avg_loss = total_loss / total
        elapsed = time.time() - start_time

        print(
            f"[Epoch {epoch + 1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Time: {elapsed:.2f}s")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    print(f"\nTest Accuracy: {100. * correct / total:.2f}%")

    torch.save(model, "model_full.pth")


if __name__ == "__main__":
    main()
