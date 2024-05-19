import torch 
import torchvision 
from torchvision import transforms 
from PIL import Image 
import pandas as pd 
import random
import os
import cv2
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载预训练的 Mask R-CNN 模型
rcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
rcnn_model.eval()

def remove_background(image_path):
    # 加载图像并转换为PyTorch tensor
    image_pil = Image.open(image_path)
    image_tensor = torchvision.transforms.functional.to_tensor(image_pil)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # 使用模型进行预测
    with torch.no_grad():
        predictions = rcnn_model(image_tensor)

    # 获取第一个对象的掩码（假设只有一个主要对象）
    mask = predictions[0]['masks'][0, 0].mul(255).byte().cpu().numpy()

    # 将图像从PIL转换为OpenCV格式（从RGB到BGR）
    image_cv = np.array(image_pil)[:, :, ::-1]

    # 应用掩码去除背景
    masked_image = cv2.bitwise_and(image_cv, image_cv, mask=mask)
    image = Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    image.save(image_path)
    return image

def find_and_enlarge_non_black_region(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image")
        return None
    
    # 定义纯黑色的阈值
    lower_black = (0, 0, 0)
    upper_black = (0, 0, 0)
    
    # 转换为HSV色彩空间，便于处理
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 创建一个黑色的掩码，然后取反，得到非黑区域
    mask = cv2.inRange(hsv, lower_black, upper_black)
    mask = cv2.bitwise_not(mask)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour
            
    if largest_contour is not None:
        # 获取最大非黑区域的边界框
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 计算放大比例，确保最长边与原图最长边等长
        img_height, img_width = image.shape[:2]
        longest_side = max(img_height, img_width)
        scale_factor = longest_side / max(h, w)
        
        # 根据边界框裁剪非黑区域
        cropped = image[y:y+h, x:x+w]
        
        # 等比例放大裁剪的非黑区域
        resized_cropped = cv2.resize(cropped, (int(w*scale_factor), int(h*scale_factor)), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(image_path, resized_cropped)

        return resized_cropped
    else:
        print("No significant non-black region found.")
        return None


# 预处理函数
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 添加batch维度
    return image

# 加载模型
def load_model():
    # 加载预训练的ResNet模型
    model = torchvision.models.resnet50(pretrained=True)
    
    # 替换全连接层以匹配训练时的结构
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 6)  # 假设您的模型输出是6个类别
    
    # 如果您在训练时冻结了其他层，并重新训练了层3和层4，这里也需要相应地设置
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.layer3.parameters():
        param.requires_grad = True
    
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # 加载训练好的权重
    device = torch.device("cpu")  # 确保在没有GPU的情况下也能正常工作

    from collections import OrderedDict

    state_dict = torch.load('model_epoch_15_0518.pth', map_location=device)

    state_dict_fixed = OrderedDict()
    for k, v in state_dict.items():
        # Remove 'module.' prefix if present
        name = k[7:] if k.startswith('module.') else k
        state_dict_fixed[name] = v

    # Load the fixed state dict
    model.load_state_dict(state_dict_fixed)

    # model.load_state_dict(torch.load('model_epoch_15_0517.pth', map_location=device))
    
    model.eval()  # 设置为评估模式
    return model

# 预测函数
def predict_image(image_path, model):
    remove_background(image_path)
    find_and_enlarge_non_black_region(image_path)
    image = preprocess_image(image_path)
    with torch.no_grad():  # 在测试阶段不计算梯度
        output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()

# 主函数
def main():
    # 读取测试集的CSV文件
    test_csv_file = 'test/_classes.csv'  # 替换为您的测试集CSV文件路径
    test_data = pd.read_csv(test_csv_file)
    for i in range(10):
        # 随机选择一个图片
        random_index = random.randint(0, len(test_data) - 1)
        image_filename = test_data.iloc[random_index, 0]
        actual_labels = test_data.iloc[random_index, 1:].values
        image_path = os.path.join('test', image_filename)  # 替换为您的测试图片文件夹路径

        #image_path = '2icu0b1h67hatu9s0bqz0o9o5_0.jpg'

        # 加载模型
        model = load_model()

        # 进行预测
        prediction = predict_image(image_path, model)

        # 打印实际类别和预测类别
        print(f'Actual labels: {actual_labels}')
        print(f'Predicted class: {prediction}')

if __name__ == '__main__':
    main()

