import torch
import torchvision
from PIL import Image
import cv2
import numpy as np
import os

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载预训练的 Mask R-CNN 模型
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()

def remove_background(image_path):
    # 加载图像并转换为PyTorch tensor
    image_pil = Image.open(image_path)
    image_tensor = torchvision.transforms.functional.to_tensor(image_pil)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # 使用模型进行预测
    with torch.no_grad():
        predictions = model(image_tensor)

    # 获取第一个对象的掩码（假设只有一个主要对象）
    mask = predictions[0]['masks'][0, 0].mul(255).byte().cpu().numpy()

    # 将图像从PIL转换为OpenCV格式（从RGB到BGR）
    image_cv = np.array(image_pil)[:, :, ::-1]

    # 应用掩码去除背景
    masked_image = cv2.bitwise_and(image_cv, image_cv, mask=mask)

    return masked_image

# 指定单张图片的路径
image_path = 'path_to_your_image.jpg'  # 请替换为你的图片路径

try:
    print(f"Processing image: {image_path}")
    # 去除背景
    result = remove_background(image_path)
    
    # 构建输出文件名（示例中直接在原文件名基础上添加'_no_bg'标识）
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{base_name}_no_bg.jpg"
    output_path = os.path.join('output', output_filename)
    
    # 确保输出目录存在
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # 保存处理后的图像
    cv2.imwrite(output_path, result)
    print(f"Processed image saved to: {output_path}")
except Exception as e:
    print(f"Error processing {image_path}: {e}")

print("Background removal for the single image completed.")