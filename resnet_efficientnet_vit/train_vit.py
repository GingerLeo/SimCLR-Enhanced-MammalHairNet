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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

batch_size = 64
val_batch_size = 64
epochs = 100
learning_rate = 0.001  
weight_decay = 8e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def gray_loader(path):
    return Image.open(path).convert("RGB")

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        return self.alpha * (1 - pt) ** self.gamma * ce_loss

criterion = FocalLoss()

def select_model(num_classes):
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model.to(device)

def train_model(model, model_name, criterion, optimizer, scheduler, num_epochs, per_tag, output_dir):
    best_acc = 0.0
    best_epoch = 0
    patience = 10
    early_stop_counter = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}\n{'-' * 10}")
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
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())
                
                scheduler.step(epoch_loss)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    print(f"[EarlyStopping] No improvement for {early_stop_counter} epoch(s).")
                    if early_stop_counter >= patience:
                        print(f"[EarlyStopping] Stopped at epoch {epoch + 1}. Best val acc: {best_acc:.4f} at epoch {best_epoch + 1}")
                        torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name}_per{per_tag}.pth"))
                        return

        torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name}_per{per_tag}.pth"))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{model_name} Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.title(f'{model_name} Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_per{per_tag}_curve.png"))
    plt.close()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)

    base_input_dir = "/data_hdd/lzh/hair_second/dataset_20250527"
    vit_output_dir = "/data_hdd/lzh/hair_second/ViT_EfficientNet/20250812/vit_hyper"

    target_dirs = [
        "2025_85_720_per25",
        "2025_85_720_per30",
        "2025_85_720_per50",
        "2025_85_720"
    ]

    for target in target_dirs:
        data_dir = os.path.join(base_input_dir, target)
        per_tag = target.split("_")[-1].replace("per", "")

        print(f"\n[INFO] 当前训练数据集：{target}")
        full_dataset = ImageFolder(
            root=data_dir,
            loader=gray_loader,
            transform=transform,
            is_valid_file=lambda x: x.endswith('.tif')
        )

        train_idx, val_idx = train_test_split(
            range(len(full_dataset)),
            test_size=0.2,
            stratify=full_dataset.targets,
            random_state=42
        )

        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)

        data_loaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
            'val': DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4)
        }
        data_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
        class_names = full_dataset.classes

        os.makedirs(vit_output_dir, exist_ok=True)
        with open(os.path.join(vit_output_dir, f"class_to_idx_per{per_tag}.json"), 'w') as f:
            json.dump(full_dataset.class_to_idx, f, indent=4)

        print(f"\n==== Training ViT on {target} ====")
        model = select_model(len(class_names))
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        train_model(model, "vit", criterion, optimizer, scheduler, epochs, per_tag, vit_output_dir)