import torch
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import datasets
from tqdm import tqdm  # 导入tqdm

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# 设置路径
base_path = '/home/lzh/hair/hair_2025/dataset/SimCLR-master/runs/Jan10_23-56-06_ubuntu20-PowerEdge-R750/'
checkpoint_path = os.path.join(base_path, 'model_best.pth.tar')
results_path = base_path

# 加载SimCLR训练的模型参数
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
state_dict = checkpoint['state_dict']

# 选择架构，假设是resnet50
model = torchvision.models.resnet50(weights=None, num_classes=25).to(device)

# 加载SimCLR预训练的backbone（去除全连接层）
for k in list(state_dict.keys()):
    if k.startswith('backbone.') and not k.startswith('backbone.fc'):
        # Remove prefix 'backbone.'
        state_dict[k[len("backbone."):]] = state_dict[k]
    del state_dict[k]

# 加载无监督学习的预训练权重
log = model.load_state_dict(state_dict, strict=False)
assert log.missing_keys == ['fc.weight', 'fc.bias']

# 训练整个网络
for name, param in model.named_parameters():
    param.requires_grad = True  # 不再冻结任何层的参数

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.0008)

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

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 根据模型需求调整图片尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 用ImageNet的均值和方差进行标准化
])

# 加载数据集
train_dataset = datasets.ImageFolder('/home/lzh/hair/hair_2025/dataset/sam_2025_85_rgb_720_per50', transform=transform)
test_dataset = datasets.ImageFolder('/home/lzh/hair/hair_2025/dataset/SAM_15_2025_100_oriname', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 准确率计算函数
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

# 训练模型
epochs = 150

# 创建文件保存路径
os.makedirs(results_path, exist_ok=True)
log_file_path = os.path.join(results_path, 'tuning_all_cos_per50_150_5-10-5_32_ori_name.txt')

# 打开文件进行写入（以追加模式打开）
log_file = open(log_file_path, 'a')

best_top1_accuracy = 0  # 记录最佳Top1准确率
best_model_state_dict = None  # 记录最佳模型的参数

# 加入学习率调度器（Warmup + CosineAnnealing）
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

# 设置 warmup 阶段和总训练 epoch
warmup_epochs = 5
total_epochs = epochs

# 定义 warmup 策略
def warmup_lambda(current_epoch):
    if current_epoch < warmup_epochs:
        return current_epoch / warmup_epochs
    else:
        return 1.0

# 创建调度器
warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_epochs - warmup_epochs))

# 将 warmup 和 cosine scheduler 结合
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

# 初始化自定义调度器
scheduler = GradualWarmupScheduler(warmup_scheduler, cosine_scheduler, warmup_epochs)

for epoch in range(epochs):
    model.train()
    top1_train_accuracy = 0
    for counter, (x_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # 前向传播
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        
        # 计算准确率
        top1 = accuracy(logits, y_batch, topk=(1,))
        top1_train_accuracy += top1[0]

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step(epoch)  # 更新学习率调度器

    top1_train_accuracy /= (counter + 1)

    # 在测试集上评估模型
    model.eval()
    top1_accuracy = 0
    top5_accuracy = 0
    with torch.no_grad():
        for counter, (x_batch, y_batch) in enumerate(test_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            logits = model(x_batch)
            top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

    top1_accuracy /= (counter + 1)
    top5_accuracy /= (counter + 1)

    result = f"Epoch {epoch+1}/{epochs}\tTop1 Train accuracy: {top1_train_accuracy.item()}%\tTop1 Test accuracy: {top1_accuracy.item()}%\tTop5 Test accuracy: {top5_accuracy.item()}%\n"
    print(result)
    log_file.write(result)

    if top1_accuracy > best_top1_accuracy:
        best_top1_accuracy = top1_accuracy
        best_model_state_dict = model.state_dict()

if best_model_state_dict is not None:
    torch.save(best_model_state_dict, os.path.join(results_path, 'tuning_all_cos_per50_best_150_5-10-5_32_ori_name.pth'))

log_file.close()
