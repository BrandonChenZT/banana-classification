import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载预训练的 Mask R-CNN 模型
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()


def remove_background(image_path):
    # 加载图像并转换为PyTorch tensor
    image_pil = Image.open(image_path)
    image_tensor = torchvision.transforms.functional.to_tensor(image_pil)
    image_tensor = image_tensor.unsqueeze(0)

    # 使用模型进行预测
    with torch.no_grad():
        predictions = model(image_tensor)

    # 获取第一个对象的掩码（假设只有一个主要对象）
    # 注意：在实际应用中，可能需要更复杂的逻辑来选择正确的掩码，特别是当图像中有多个对象时。
    mask = predictions[0]['masks'][0, 0].mul(255).byte().cpu().numpy()

    # 将图像从PIL转换为OpenCV格式（从RGB到BGR）
    image_cv = np.array(image_pil)[:, :, ::-1]

    # 应用掩码去除背景
    # 注意：这里直接使用掩码进行位与操作可能不完美，特别是对于透明或半透明掩码处理。
    # 对于复杂背景去除，可能需要进一步的后处理。
    masked_image = cv2.bitwise_and(image_cv, image_cv, mask=mask)

    # 返回去除背景后的图像
    return masked_image


# 读取CSV文件
csv_file = 'banana-classification/test/_classes.csv'  # 请替换为你的CSV文件路径
data = pd.read_csv(csv_file)

# 确保输出目录存在
output_dir = 'banana-classification/test/background_removed_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历CSV文件中的每一行，对每张图片应用去背景操作并保存
for index, row in data.iterrows():
    image_path = 'banana-classification/test/' + row['filename']
    try:
        print(f"Processing image: {image_path}")
        # 去除背景
        result = remove_background(image_path)
        
        # 构建新的文件名，保存到输出目录
        base_name = os.path.basename(image_path)
        new_file_path = os.path.join(output_dir, base_name)
        cv2.imwrite(new_file_path, result)
        print(f"Processed image saved to: {new_file_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

print("Background removal process completed.")