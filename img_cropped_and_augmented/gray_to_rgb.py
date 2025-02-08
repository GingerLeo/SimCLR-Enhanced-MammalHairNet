import os
import cv2
from tqdm import tqdm

# 输入文件夹和输出文件夹路径
input_folder = "/home/lzh/hair/hair_2025/dataset/SAM_per85_2025_filter_square_aug720"
output_folder = "/home/lzh/hair/hair_2025/dataset/sam_2025_85_rgb_720"

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

def process_image(input_path, output_path):
    # 读取灰度图像
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"无法读取图像: {input_path}")
        return

    # 将灰度图像复制为三通道
    image_rgb = cv2.merge([image, image, image])

    # 调整大小为 [450, 450]
    image_resized = cv2.resize(image_rgb, (450, 450), interpolation=cv2.INTER_AREA)

    # 保存处理后的图像为 .jpg 格式
    output_path = os.path.splitext(output_path)[0] + ".jpg"
    cv2.imwrite(output_path, image_resized)

def main():
    # 遍历所有子文件夹和文件
    for root, _, files in os.walk(input_folder):
        for file in tqdm(files):
            if file.lower().endswith(".tif"):
                input_path = os.path.join(root, file)

                # 构建对应的输出路径
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, file)

                # 处理图像
                process_image(input_path, output_path)

if __name__ == "__main__":
    main()
