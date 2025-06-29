import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# =============== 随机种子 ================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# =============== 目录和参数 ================
data_dir = "/data_hdd/lzh/hair/dataset_20250527/2025_85_720/"
batch_size = 32
epochs = 50
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "/data_hdd/lzh/hair/ViT_EfficientNet/20250527/models_3/all_720"
os.makedirs(output_dir, exist_ok=True)

# =============== 数据增强 ================
data_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
}

def gray_loader(path):
    return Image.open(path).convert("RGB")

# =============== 数据加载 ================
full_dataset = ImageFolder(
    root=data_dir,
    loader=gray_loader,
    transform=data_transforms['train'],
    is_valid_file=lambda x: x.endswith('.tif')
)

# 输出 class_to_idx 并保存，仅输出一次
if __name__ == '__main__':
    print("\n[INFO] class_to_idx mapping:")
    print(json.dumps(full_dataset.class_to_idx, indent=4))
    with open(os.path.join(output_dir, "class_to_idx.json"), 'w') as f:
        json.dump(full_dataset.class_to_idx, f, indent=4)

train_indices, val_indices = train_test_split(
    range(len(full_dataset)),
    test_size=0.2,
    stratify=full_dataset.targets,
    random_state=42
)

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

data_loaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
}
data_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = full_dataset.classes

# =============== Focal Loss ================
def focal_loss(alpha=0.25, gamma=2.0):
    class FocalLoss(nn.Module):
        def __init__(self, alpha, gamma):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            ce_loss = nn.CrossEntropyLoss()(inputs, targets)
            pt = torch.exp(-ce_loss)
            return self.alpha * (1 - pt) ** self.gamma * ce_loss

    return FocalLoss(alpha, gamma)

criterion = focal_loss(alpha=0.25, gamma=2.0)

# =============== 模型选择器 ================
def select_model(model_name, num_classes):
    if model_name == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet":
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "vit":
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError("Unknown model: " + model_name)

    return model.to(device)

# =============== 训练过程 ================
def train_model(model, model_name, criterion, optimizer, scheduler, num_epochs=epochs):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in tqdm(data_loaders[phase], desc=f"{phase} Epoch {epoch+1}"):
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

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
                print(f"Learning rate after step {epoch+1}: {scheduler.get_last_lr()[0]}")
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)

    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Acc')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Val Acc')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_{model_name}_720_32_50_001.png"))
    return model

# =============== 主函数 ================
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)

    for model_name in ["resnet50", "efficientnet", "vit"]:
        print(f"\n==== Training {model_name.upper()} ====")
        model = select_model(model_name, len(class_names))
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        trained_model = train_model(model, model_name, criterion, optimizer, scheduler)
        torch.save(trained_model.state_dict(),
                   os.path.join(output_dir, f"training_{model_name}_720_32_50_001.pth"))
