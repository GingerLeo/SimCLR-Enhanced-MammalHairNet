import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import DatasetFolder
from PIL import Image
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Paths to data directories
data_dir = "/home/lzh/hair/hair_2025/SAM_per85_2025_filter_square_aug720"
output_dir = "/home/lzh/hair/hair_2025/python/fold_5_32_30_001"

# Hyperparameters
batch_size = 32
epochs = 30
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations for training and validation
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define custom loader for grayscale images
def gray_loader(path):
    return Image.open(path).convert("RGB")

# Create dataset
full_dataset = DatasetFolder(
    root=data_dir,
    loader=gray_loader,
    extensions=('.tif', '.tiff'),
    transform=data_transforms
)

# Get class names
class_names = full_dataset.classes

# Define the model
def initialize_model(num_classes):
    from torchvision.models import ResNet50_Weights
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# Define Focal Loss
def focal_loss(alpha=0.25, gamma=2.0):
    class FocalLoss(nn.Module):
        def __init__(self, alpha, gamma):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            ce_loss = nn.CrossEntropyLoss()(inputs, targets)
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            return focal_loss

    return FocalLoss(alpha, gamma)

criterion = focal_loss(alpha=0.25, gamma=2.0)

# Define optimizer and scheduler
def get_optimizer_and_scheduler(model):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    return optimizer, scheduler

# Training loop with visualization
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=epochs, fold_idx=0):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} Epoch {epoch+1}"):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
                print(f"Learning rate after step {epoch+1}: {scheduler.get_last_lr()[0]}")
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())
                scheduler.step(epoch_loss)

                # Save the best model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()

    # Save model and training logs for the fold
    fold_dir = os.path.join(output_dir, str(fold_idx))
    os.makedirs(fold_dir, exist_ok=True)
    torch.save(best_model_wts, os.path.join(fold_dir, f"best_model_fold_{fold_idx}.pth"))

    plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, "training_progress.png"))
    plt.close()

    return best_acc


# Perform 10-fold cross-validation
def cross_validate_model():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_acc = []  # List to store accuracies for each fold
    best_fold_idx = -1  # Variable to store the best fold index
    best_acc = 0.0  # Variable to store the best accuracy

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(np.zeros(len(full_dataset)), full_dataset.targets)):
        print(f"Fold {fold_idx + 1}")
        
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        }

        dataset_sizes = {
            'train': len(train_dataset),
            'val': len(val_dataset)
        }

        model = initialize_model(len(class_names)).to(device)
        optimizer, scheduler = get_optimizer_and_scheduler(model)

        # Train the model and get the accuracy for this fold
        fold_acc = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=epochs, fold_idx=fold_idx)
        
        all_acc.append(fold_acc.cpu().item())  # Append the fold accuracy to the list
        
        # Update the best fold if current fold accuracy is higher
        if fold_acc > best_acc:
            best_acc = fold_acc
            best_fold_idx = fold_idx

    print(f"Best Fold: {best_fold_idx + 1} with Accuracy: {best_acc:.4f}")
    print(f"Mean Accuracy: {np.mean(all_acc):.4f}, Std: {np.std(all_acc):.4f}")

# Main function
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    cross_validate_model()
 