import torch
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

# ---------------- 设置随机种子 ---------------- #
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------- 设置设备 ---------------- #
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
print("Using device:", device)

# ---------------- 设置路径 ---------------- #
data_root = '/data_hdd/lzh/hair/dataset_20250527/2025_85_720'
results_path = '/data_hdd/lzh/hair/ViT_EfficientNet/20250531/sim_33_ori'
os.makedirs(results_path, exist_ok=True)
best_model_path = os.path.join(results_path, 'finetuning_sim_33_720.pth')
log_file_path = os.path.join(results_path, 'finetuning_sim_33_720_log.txt')

# ---------------- 数据预处理 ---------------- #
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- 加载数据集并保存 class_to_idx ---------------- #
full_dataset = ImageFolder(root=data_root, transform=transform)
class_to_idx_path = os.path.join(results_path, 'class_to_idx.json')
with open(class_to_idx_path, 'w') as f:
    json.dump(full_dataset.class_to_idx, f, indent=4)
print("Class to index mapping saved to class_to_idx.json:", full_dataset.class_to_idx)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# ---------------- 加载SimCLR预训练模型 ---------------- #
checkpoint_path = os.path.join(results_path, 'sim33_ori.pth.tar')
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
state_dict = checkpoint['state_dict']

model = torchvision.models.resnet50(weights=None, num_classes=25).to(device)

for k in list(state_dict.keys()):
    if k.startswith('backbone.') and not k.startswith('backbone.fc'):
        state_dict[k[len("backbone."):]] = state_dict[k]
    del state_dict[k]

log = model.load_state_dict(state_dict, strict=False)
assert log.missing_keys == ['fc.weight', 'fc.bias']

for param in model.parameters():
    param.requires_grad = True

# ---------------- 优化器与损失函数 ---------------- #
optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=8e-4)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss

criterion = FocalLoss()

# ---------------- Warmup + Cosine Scheduler ---------------- #
warmup_epochs = 5
total_epochs = 150

# warmup lambda function
def warmup_lambda(current_epoch):
    if current_epoch < warmup_epochs:
        return current_epoch / warmup_epochs
    else:
        return 1.0

warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_epochs - warmup_epochs))

class GradualWarmupScheduler:
    def __init__(self, warmup_scheduler, cosine_scheduler, warmup_epochs):
        self.warmup_scheduler = warmup_scheduler
        self.cosine_scheduler = cosine_scheduler
        self.warmup_epochs = warmup_epochs

    def step(self, current_epoch):
        if current_epoch < self.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step(current_epoch - self.warmup_epochs)

scheduler = GradualWarmupScheduler(warmup_scheduler, cosine_scheduler, warmup_epochs)

# ---------------- 准确率函数 ---------------- #
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# ---------------- 训练循环 ---------------- #
best_acc = 0.0
log_file = open(log_file_path, 'a')
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(total_epochs):
    model.train()
    running_loss = 0.0
    top1_train = 0

    train_loader_tqdm = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{total_epochs}", ncols=100)
    for x_batch, y_batch in train_loader_tqdm:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1 = accuracy(logits, y_batch, topk=(1,))
        top1_train += acc1[0]
        running_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=loss.item(), acc=acc1[0].item())

    avg_train_acc = top1_train / len(train_loader)
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_acc.item())

    model.eval()
    val_loss = 0.0
    val_top1 = 0
    val_top5 = 0

    val_loader_tqdm = tqdm(val_loader, desc=f"[Val  ] Epoch {epoch+1}/{total_epochs}", ncols=100)
    with torch.no_grad():
        for x_batch, y_batch in val_loader_tqdm:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            acc1, acc5 = accuracy(logits, y_batch, topk=(1, 5))
            val_loss += loss.item()
            val_top1 += acc1[0]
            val_top5 += acc5[0]
            val_loader_tqdm.set_postfix(loss=loss.item(), top1=acc1[0].item(), top5=acc5[0].item())

    avg_val_loss = val_loss / len(val_loader)
    avg_val_top1 = val_top1 / len(val_loader)
    avg_val_top5 = val_top5 / len(val_loader)
    val_losses.append(avg_val_loss)
    val_accuracies.append(avg_val_top1.item())

    scheduler.step(epoch)

    current_lr = optimizer.param_groups[0]['lr']

    result = (f"Epoch {epoch+1}/{total_epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Top1 Acc: {avg_train_acc.item():.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Top1 Acc: {avg_val_top1.item():.2f}%, Top5 Acc: {avg_val_top5.item():.2f}% | "
              f"LR: {current_lr:.6f}\n")
    print(result)
    log_file.write(result)

    if avg_val_top1 > best_acc:
        best_acc = avg_val_top1
        torch.save(model.state_dict(), best_model_path)

log_file.close()

# ---------------- 绘制训练曲线 ---------------- #
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].plot(train_accuracies, label='Train Acc')
axs[0].plot(val_accuracies, label='Val Acc')
axs[0].set_title('Accuracy Curve')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Top1 Accuracy (%)')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(train_losses, label='Train Loss')
axs[1].plot(val_losses, label='Val Loss')
axs[1].set_title('Loss Curve')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(results_path, 'sim33_720_training_curves.png'))
plt.close()
