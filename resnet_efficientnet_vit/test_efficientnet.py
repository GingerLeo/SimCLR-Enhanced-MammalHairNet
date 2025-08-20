import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ===== 1. 设置路径 =====
efficientnet_dir = "/data_hdd/lzh/hair_second/ViT_EfficientNet/20250808/efficientnet_hyper"
test_dir = "/data_hdd/lzh/hair_second/dataset_20250527/2025_15_100"
output_csv = "evaluation_results_efficientnet.csv"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===== 2. 图像预处理 =====
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def gray_loader(path):
    return Image.open(path).convert("RGB")

# ===== 3. 加载测试集 =====
test_dataset = ImageFolder(root=test_dir, transform=transform, loader=gray_loader)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)
num_classes = len(test_dataset.classes)

# ===== 4. 加载模型 =====
def load_efficientnet(pth_path):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    return model.to(device)

# ===== 5. 测试函数 =====
def evaluate_model(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating EfficientNet"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ===== 6. 遍历所有模型并评估 =====
results = []
for file in os.listdir(efficientnet_dir):
    if file.endswith(".pth"):
        pth_path = os.path.join(efficientnet_dir, file)
        print(f"\n🔍 正在测试 EfficientNet 模型: {file}")
        try:
            model = load_efficientnet(pth_path)
            acc = evaluate_model(model)
            results.append({"model_type": "efficientnet", "file_name": file, "top1_acc": round(acc, 4)})
        except Exception as e:
            print(f"[跳过] {file} 加载失败: {e}")

# ===== 7. 保存结果 =====
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"\n✅ EfficientNet 模型测试完成，结果已保存至：{output_csv}")
