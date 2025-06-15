import torch
import random
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib.patches import Patch
import os

# =============== 随机种子 ================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ---------------- 参数 ---------------- #
model_path = "/data_hdd/lzh/hair/ViT_EfficientNet/20250531/sim_33_ori/finetuning_sim_33_720.pth"
data_root = "/data_hdd/lzh/hair/dataset_20250527/2025_15_100"
output_dir = "/data_hdd/lzh/hair/ViT_EfficientNet/20250605/pic_tree"
os.makedirs(output_dir, exist_ok=True)
distance_type = "cosine"
batch_size = 32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ---------------- 自定义标签映射 ---------------- #
custom_label_map = {
    "Asian_Badger": 0, "Asian_Black_bear": 1, "Blue_sheep": 2, "Bornean_orangutan": 3,
    "Brown_bear": 4, "Chimpanzee": 5, "Clouded_leopard": 6, "Eurasian_Otter": 7,
    "Forest_Musk_deer": 8, "Grey_wolf": 9, "Jungle_cat": 10, "Leopard": 11,
    "Leopard_cat": 12, "Long_tailed_goral": 13, "Pallas_s_cat": 14, "Raccoon_dog": 15,
    "Red_fox": 16, "Siberian_Musk_deer": 17, "Siberian_Roe_deer": 18, "Sika_deer": 19,
    "Tiger": 20, "Wapiti": 21, "Water_deer": 22, "Wild_boar": 23, "Yellow_throated_marten": 24
}
idx_to_class = {v: k for k, v in custom_label_map.items()}
num_classes = len(idx_to_class)

# ---------------- 群组颜色分配 ---------------- #
groups = {
    "Carnivora": ["Asian_Badger", "Asian_Black_bear", "Brown_bear", "Clouded_leopard", "Eurasian_Otter",
                  "Grey_wolf", "Jungle_cat", "Leopard", "Leopard_cat", "Pallas_s_cat", "Raccoon_dog",
                  "Red_fox", "Tiger", "Yellow_throated_marten"],
    "Artiodactyla": ["Blue_sheep", "Forest_Musk_deer", "Long_tailed_goral", "Siberian_Musk_deer",
                     "Siberian_Roe_deer", "Sika_deer", "Wapiti", "Water_deer", "Wild_boar"],
    "Primates": ["Bornean_orangutan", "Chimpanzee"]
}
group_colors = {
    "Carnivora": "red",
    "Artiodactyla": "green",
    "Primates": "blue"
}
label_color_map = {}
label_group_map = {}
for group, species_list in groups.items():
    for species in species_list:
        label_color_map[species] = group_colors[group]
        label_group_map[species] = group

# ---------------- 数据变换 ---------------- #
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- 加载数据 ---------------- #
dataset = ImageFolder(root=data_root, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# ---------------- 加载模型 ---------------- #
backbone = torchvision.models.resnet50(weights=None)
backbone.fc = nn.Linear(2048, num_classes)
state_dict = torch.load(model_path, map_location=device)
backbone.load_state_dict(state_dict)
backbone.fc = nn.Identity()
backbone = backbone.to(device)
backbone.eval()

# ---------------- 提取每类特征均值 ---------------- #
class_features = {i: [] for i in range(num_classes)}
with torch.no_grad():
    for images, labels in data_loader:
        images = images.to(device)
        features = backbone(images)
        for f, label in zip(features, labels):
            class_features[label.item()].append(f.cpu().numpy())

mean_features = []
for i in range(num_classes):
    feats = np.stack(class_features[i], axis=0)
    mean_vec = feats.mean(axis=0)
    mean_features.append(mean_vec)
mean_features = np.stack(mean_features, axis=0)

# ---------------- 距离矩阵和聚类 ---------------- #
distance_vector = pdist(mean_features, metric=distance_type)
linked = linkage(distance_vector, method='average')
distance_matrix = squareform(distance_vector)  # 用于 heatmap
labels = [idx_to_class[i] for i in range(num_classes)]

# ---------------- 构建聚类索引对应叶子节点的映射 ---------------- #
def build_cluster_to_labels(linkage_matrix, labels):
    cluster_to_leaves = {i: [i] for i in range(len(labels))}
    n = len(labels)
    for cluster_id, (left, right, _, _) in enumerate(linkage_matrix):
        cluster_to_leaves[n + cluster_id] = cluster_to_leaves[int(left)] + cluster_to_leaves[int(right)]
    return cluster_to_leaves

# ---------------- 生成颜色函数 ---------------- #
cluster_to_leaves = build_cluster_to_labels(linked, labels)

def majority_group_color(cluster_id):
    leaf_indices = cluster_to_leaves[cluster_id]
    leaf_labels = [labels[i] for i in leaf_indices]
    group_count = {}
    for lbl in leaf_labels:
        grp = label_group_map.get(lbl)
        if grp:
            group_count[grp] = group_count.get(grp, 0) + 1
    if not group_count:
        return "gray"
    majority_group = max(group_count, key=group_count.get)
    return group_colors[majority_group]

# ---------------- 绘制染色树状图 ---------------- #
plt.figure(figsize=(14, 6))
dendrogram(linked,
           labels=labels,
           leaf_rotation=90,
           leaf_font_size=10,
           link_color_func=majority_group_color)
plt.title(f"Class Hierarchy Dendrogram ({distance_type}) with Link Colors by Dominant Group")

# 图例
legend_elements = [Patch(facecolor=color, label=group) for group, color in group_colors.items()]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'dendrogram_majority_colored_{distance_type}.png'))
plt.close()

# ---------------- 热图 ---------------- #
plt.figure(figsize=(14, 12))
sns.heatmap(distance_matrix,
            xticklabels=labels,
            yticklabels=labels,
            cmap='viridis', square=True, annot=True, fmt=".2f")
plt.title(f'Class Distance Matrix ({distance_type})')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'distance_heatmap_{distance_type}.png'))
plt.close()
