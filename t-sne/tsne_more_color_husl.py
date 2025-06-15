import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
import seaborn as sns
import plotly.express as px
import pandas as pd

# ---------------- 参数 ---------------- #
model_path = "/data_hdd/lzh/hair/ViT_EfficientNet/20250531/sim_33_ori/finetuning_sim_33_720.pth"
data_root = "/data_hdd/lzh/hair/dataset_20250527/2025_15_100"
output_dir = "/data_hdd/lzh/hair/ViT_EfficientNet/20250605/tsne_vis_more_color_husl"
os.makedirs(output_dir, exist_ok=True)
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自定义标签名
label_map = {
    0: "Asian_Badger", 1: "Asian_Black_bear", 2: "Blue_sheep", 3: "Bornean_orangutan", 4: "Brown_bear",
    5: "Chimpanzee", 6: "Clouded_leopard", 7: "Eurasian_Otter", 8: "Forest_Musk_deer", 9: "Grey_wolf",
    10: "Jungle_cat", 11: "Leopard", 12: "Leopard_cat", 13: "Long_tailed_goral", 14: "Pallas_s_cat",
    15: "Raccoon_dog", 16: "Red_fox", 17: "Siberian_Musk_deer", 18: "Siberian_Roe_deer", 19: "Sika_deer",
    20: "Tiger", 21: "Wapiti", 22: "Water_deer", 23: "Wild_boar", 24: "Yellow_throated_marten"
}

# ---------------- 数据预处理 ---------------- #
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder(root=data_root, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
class_names = [label_map[i] for i in range(len(label_map))]

# ---------------- 加载模型 ---------------- #
backbone = torchvision.models.resnet50(weights=None)
backbone.fc = nn.Linear(2048, 25)  # 模型结构匹配
state_dict = torch.load(model_path, map_location=device)
backbone.load_state_dict(state_dict)
backbone.fc = nn.Identity()  # 去掉分类头
backbone = backbone.to(device)
backbone.eval()

# ---------------- 提取特征 ---------------- #
features = []
labels = []

with torch.no_grad():
    for imgs, lbls in data_loader:
        imgs = imgs.to(device)
        outs = backbone(imgs)
        features.append(outs.cpu().numpy())
        labels.extend(lbls.numpy())

features = np.concatenate(features, axis=0)
labels = np.array(labels)

# ---------------- t-SNE 降维 (2D) ---------------- #
tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
embed_2d = tsne_2d.fit_transform(features)

# 使用 seaborn 的调色板（高区分度）
palette = sns.color_palette("husl", len(class_names))

# ---------------- 绘制 2D t-SNE ---------------- #
plt.figure(figsize=(12, 10))
for i in range(len(class_names)):
    idxs = (labels == i)
    plt.scatter(embed_2d[idxs, 0], embed_2d[idxs, 1], label=class_names[i], s=10, color=palette[i])
plt.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.25, 1.0))
plt.title("t-SNE Visualization (2D)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "tsne_2d_colored.png"), dpi=300)
plt.close()

# ---------------- t-SNE 降维 (3D) ---------------- #
from mpl_toolkits.mplot3d import Axes3D

tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
embed_3d = tsne_3d.fit_transform(features)

# ---------------- 绘制 3D t-SNE ---------------- #
# 创建 DataFrame
df = pd.DataFrame(embed_3d, columns=["x", "y", "z"])
df['label'] = [class_names[i] for i in labels]

# 绘制交互式图
fig = px.scatter_3d(df, x='x', y='y', z='z',
                    color='label',
                    title='t-SNE-Based Visualization of Hair Scale Feature Embeddings on the Test Set of 25 Mammalian Species Using a Fine-Tuned SimCLR Model',
                    width=1800, height= 1400)

fig.update_traces(marker=dict(size=4))  # 调整点大小
fig.write_html(os.path.join(output_dir, "tsne_3d_interactive.html"))
print("交互式图已保存为 HTML 文件")
