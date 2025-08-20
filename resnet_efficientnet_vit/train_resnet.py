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
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

set_seed(42)

# ===== 超参数 =====
batch_size = 128
val_batch_size = 64
epochs = 100
learning_rate = 5e-5
weight_decay = 8e-4
warmup_epochs = 5
patience = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===== 图像预处理 =====
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
def gray_loader(path): return Image.open(path).convert("RGB")

# ===== Focal Loss =====
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__(); self.alpha=alpha; self.gamma=gamma
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        return self.alpha * (1-pt)**self.gamma * ce_loss
criterion = FocalLoss()

# ===== 模型选择 =====
def select_model(num_classes):
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

# ===== 学习率调度器 =====
class GradualWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs):
        self.optimizer=optimizer; self.warmup_epochs=warmup_epochs; self.total_epochs=total_epochs
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    def step(self, current_epoch):
        if current_epoch < self.warmup_epochs:
            warmup_lr = learning_rate * (current_epoch+1)/self.warmup_epochs
            for pg in self.optimizer.param_groups: pg['lr'] = warmup_lr
        else:
            self.cosine_scheduler.step(current_epoch - self.warmup_epochs)

# ===== 训练函数 =====
def train_model(model, criterion, optimizer, scheduler, num_epochs, per_tag, output_dir):
    best_acc, best_epoch, early_stop_counter = 0.0, 0, 0
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}\n{'-'*10}")
        for phase in ['train', 'val']:
            model.train() if phase=='train' else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in tqdm(data_loaders[phase], desc=f"{phase} Epoch {epoch+1}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase=='train':
                        loss.backward(); optimizer.step()
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double()/data_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase=='train':
                train_losses.append(epoch_loss); train_accuracies.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss); val_accuracies.append(epoch_acc.item())
                if epoch_acc > best_acc:
                    best_acc, best_epoch, early_stop_counter = epoch_acc, epoch, 0
                else:
                    early_stop_counter += 1
                    print(f"[EarlyStopping] No improvement for {early_stop_counter} epoch(s).")
                    if early_stop_counter>=patience:
                        print(f"[EarlyStopping] Stop at epoch {epoch+1}. Best val acc: {best_acc:.4f} at epoch {best_epoch+1}")
                        scheduler.step(epoch)
                        torch.save(model.state_dict(), os.path.join(output_dir, f"resnet50_per{per_tag}.pth"))
                        return

        scheduler.step(epoch)

    torch.save(model.state_dict(), os.path.join(output_dir, f"resnet50_per{per_tag}.pth"))

    # 保存曲线
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1); plt.plot(train_losses,label='Train Loss'); plt.plot(val_losses,label='Val Loss')
    plt.title('ResNet50 Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.subplot(1,2,2); plt.plot(train_accuracies,label='Train Acc'); plt.plot(val_accuracies,label='Val Acc')
    plt.title('ResNet50 Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"resnet50_per{per_tag}_curve.png")); plt.close()

# ===== 主程序 =====
if __name__=='__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)

    base_input_dir="/data_hdd/lzh/hair_second/dataset_20250527"
    output_dir="/data_hdd/lzh/hair_second/ViT_EfficientNet/20250807/resnet50_hyper"
    target_dirs=["2025_85_720_per25","2025_85_720_per30","2025_85_720_per50","2025_85_720"]

    os.makedirs(output_dir, exist_ok=True)

    for target in target_dirs:
        data_dir=os.path.join(base_input_dir, target); per_tag=target.split("_")[-1].replace("per","")
        print(f"\n[INFO] 当前训练数据集：{target}")
        full_dataset=ImageFolder(root=data_dir,loader=gray_loader,transform=transform,is_valid_file=lambda x:x.endswith('.tif'))
        train_idx,val_idx=train_test_split(range(len(full_dataset)),test_size=0.2,stratify=full_dataset.targets,random_state=42)
        train_dataset, val_dataset=Subset(full_dataset,train_idx), Subset(full_dataset,val_idx)

        data_loaders={'train': DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4),
                      'val': DataLoader(val_dataset,batch_size=val_batch_size,shuffle=False,num_workers=4)}
        data_sizes={'train': len(train_dataset), 'val': len(val_dataset)}
        class_names=full_dataset.classes

        with open(os.path.join(output_dir, f"class_to_idx_per{per_tag}.json"), 'w') as f: json.dump(full_dataset.class_to_idx, f, indent=4)

        print(f"\n==== Training RESNET50 on {target} ====")
        model=select_model(len(class_names))
        optimizer=optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler=GradualWarmupScheduler(optimizer,warmup_epochs,warmup_epochs+epochs)
        train_model(model, criterion, optimizer, scheduler, epochs, per_tag, output_dir)
