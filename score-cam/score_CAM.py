import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision import models
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = models.resnet50(weights=None, num_classes=25)
ckpt_path = '/data_hdd/lzh/hair/ViT_EfficientNet/20250531/sim_33_ori/finetuning_sim_33_720.pth'
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.to(device)
model.eval()

# 设置目标层
target_layers = [model.layer4[-1]]
cam = ScoreCAM(model=model, target_layers=target_layers)

# 输入输出路径
input_root = '/data_hdd/lzh/hair/dataset_20250527/2025_15_100/'
output_root = '/data_hdd/lzh/hair/score_cam_20250610'
os.makedirs(output_root, exist_ok=True)

# 获取所有类别子文件夹
class_dirs = [d for d in sorted(os.listdir(input_root)) if os.path.isdir(os.path.join(input_root, d))]

# 初始化进度条（总图像数为 25 类 × 100 图）
total_images = len(class_dirs) * 100
pbar = tqdm(total=total_images, desc="Processing images", unit="img")

for class_name in class_dirs:
    class_dir = os.path.join(input_root, class_name)
    output_class_dir = os.path.join(output_root, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    # 只处理 .tif 图像，最多取前 100 张
    img_files = [f for f in sorted(os.listdir(class_dir)) if f.endswith('.tif')][:100]

    for img_file in img_files:
        img_path = os.path.join(class_dir, img_file)
        try:
            # 读取 .tif 图像
            bgr_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr_img is None:
                raise ValueError("Image load failed (None)")

            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            rgb_img_resized = cv2.resize(rgb_img, (224, 224))
            rgb_float = np.float32(rgb_img_resized) / 255.0

            # 预处理
            input_tensor = preprocess_image(rgb_float, mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])
            input_tensor = input_tensor.to(device)

            # 获取目标类别
            target_class = int(class_name) if class_name.isdigit() else None
            targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

            # 计算 Score-CAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
            visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

            # 保存结果
            out_name = os.path.splitext(img_file)[0] + ".jpg"
            out_path = os.path.join(output_class_dir, out_name)
            cv2.imwrite(out_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

            pbar.update(1)
        except Exception as e:
            print(f"[!] Failed: {img_path} due to {e}")
            pbar.update(1)

pbar.close()
