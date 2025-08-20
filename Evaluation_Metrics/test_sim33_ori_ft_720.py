import torch
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import os
import pandas as pd

# =============== 随机种子 ================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 设置计算设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 测试集路径
test_dir = "/data_hdd/lzh/hair/dataset_20250527/2025_15_100/"
output_dir = "/data_hdd/lzh/hair/ViT_EfficientNet/20250610/sim_33_ori/test_720" 
os.makedirs(output_dir, exist_ok=True)  # 创建输出文件夹

# 预处理操作
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载测试集
test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 载入模型
model = resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(test_dataset.classes))
model.load_state_dict(torch.load("/data_hdd/lzh/hair/ViT_EfficientNet/20250610/sim33_tuning_all_720_best_150_5-10-5_32.pth"))
model = model.to(device)
model.eval()

# 初始化变量
all_labels = []
all_preds = []
all_probs = []

# 测试并收集结果
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)  # 计算概率
        top3_probs, top3_preds = torch.topk(probs, 3, dim=1)  # 获取Top 3预测

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(top3_preds.cpu().numpy())
        all_probs.extend(top3_probs.cpu().numpy())

# 计算混淆矩阵
cm = confusion_matrix(all_labels, [p[0] for p in all_preds])
class_names = test_dataset.classes

# 计算混淆矩阵
cm = confusion_matrix(all_labels, [p[0] for p in all_preds])
class_names = test_dataset.classes

# 绘制混淆矩阵
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",  # 整数格式
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
    linecolor="black"
)

# 添加标题和标签
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# 保存为矢量图
cm_fig_path = os.path.join(output_dir, "confusion_matrix.svg")
plt.savefig(cm_fig_path, format="svg", bbox_inches="tight")
plt.close()


# 输出分类统计
metrics = precision_recall_fscore_support(all_labels, [p[0] for p in all_preds], average=None, labels=range(len(class_names)))
overall_accuracy = accuracy_score(all_labels, [p[0] for p in all_preds])

# 计算每个类别的 Accuracy
class_accuracy = []
for i in range(len(class_names)):
    correct = cm[i, i]
    total = cm[i].sum()
    class_accuracy.append(correct / total if total > 0 else 0)

# 输出分类统计
results = []
for i, class_name in enumerate(class_names):
    results.append({
        "Class": class_name,
        "Accuracy": class_accuracy[i],  # 使用每个类别的准确率
        "Precision": metrics[0][i],
        "Recall": metrics[1][i],
        "F1 Score": metrics[2][i]
    })

# 添加全局指标
overall_metrics = {
    "Class": "Overall",
    "Accuracy": overall_accuracy,  # 全局准确率
    "Precision": np.mean(metrics[0]),
    "Recall": np.mean(metrics[1]),
    "F1 Score": np.mean(metrics[2])
}
results_df = pd.DataFrame(results)
results_df = pd.concat([results_df, pd.DataFrame([overall_metrics])], ignore_index=True)

# 保存结果
results_path = os.path.join(output_dir, "classification_metrics.csv")
results_df.to_csv(results_path, index=False)

# 输出Top 3表格
top1_correct = sum([label in pred[:1] for label, pred in zip(all_labels, all_preds)])
top2_correct = sum([label in pred[:2] for label, pred in zip(all_labels, all_preds)])
top3_correct = sum([label in pred[:3] for label, pred in zip(all_labels, all_preds)])
no_match = len(all_labels) - top3_correct

top_results = {
    "Top 1 Correct": top1_correct,
    "Top 2 Correct": top2_correct,
    "Top 3 Correct": top3_correct,
    "No Match": no_match
}
top_results_path = os.path.join(output_dir, "top_results.csv")
pd.DataFrame([top_results]).to_csv(top_results_path, index=False)

# 输出每张图片的Top 3预测及概率
image_results = []
for i, (label, preds, probs) in enumerate(zip(all_labels, all_preds, all_probs)):
    image_results.append({
        "True Label": class_names[label],
        "Top 1 Prediction": class_names[preds[0]],
        "Top 1 Probability": probs[0],
        "Top 2 Prediction": class_names[preds[1]],
        "Top 2 Probability": probs[1],
        "Top 3 Prediction": class_names[preds[2]],
        "Top 3 Probability": probs[2]
    })
image_results_df = pd.DataFrame(image_results)
image_results_path = os.path.join(output_dir, "image_results.csv")
image_results_df.to_csv(image_results_path, index=False)