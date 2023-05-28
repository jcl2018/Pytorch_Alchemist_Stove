import torch

from tqdm import tqdm

from option import num_epochs, device
from dataset import train_dataset, test_dataset, train_loader, test_loader
from model import mlp_model
from metrics import cross_entropy_loss
from utilities import adam_optimizer

# Step5: training/testing runner
for epoch in range(num_epochs):
    # 5.1 Training loop
    mlp_model.train()
    train_loss = 0.0
    train_correct = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = mlp_model(images)
        loss = cross_entropy_loss(outputs, labels)

        # Backward pass and optimization
        adam_optimizer.zero_grad()
        loss.backward()
        adam_optimizer.step()

        # Compute training metrics
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()

    # Compute average training metrics
    train_loss = train_loss / len(train_dataset)
    train_accuracy = train_correct / len(train_dataset)

    # 5.2 Evaluate the model on the test set
    mlp_model.eval()
    test_loss = 0.0
    test_correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = mlp_model(images)
            loss = cross_entropy_loss(outputs, labels)

            # Compute test metrics
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()

    # 5.3 Metrics and evaluation
    # Compute average test metrics
    test_loss = test_loss / len(test_dataset)
    test_accuracy = test_correct / len(test_dataset)

    # Print the training progress
    print(f"Epoch {epoch + 1}/{num_epochs}: "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
