from flask import Flask, request, render_template,jsonify
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import pandas as pd
import os
import cv2
import numpy as np

app = Flask(__name__)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # 加载预训练的 Mask R-CNN 模型
# rcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
# rcnn_model.eval()

# def remove_background(image_path):
    # 加载图像并转换为PyTorch tensor
    # image_pil = Image.open(image_path)
    # image_tensor = torchvision.transforms.functional.to_tensor(image_pil)
    # image_tensor = image_tensor.unsqueeze(0).to(device)

    # # 使用模型进行预测
    # with torch.no_grad():
    #     predictions = rcnn_model(image_tensor)

    # # 获取第一个对象的掩码（假设只有一个主要对象）
    # mask = predictions[0]['masks'][0, 0].mul(255).byte().cpu().numpy()

    # # 将图像从PIL转换为OpenCV格式（从RGB到BGR）
    # image_cv = np.array(image_pil)[:, :, ::-1]

    # # 应用掩码去除背景
    # masked_image = cv2.bitwise_and(image_cv, image_cv, mask=mask)
    # image = Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    # image.save(image_path)
    # return image

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

    state_dict = torch.load('model_epoch_15_0517.pth', map_location=device)

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
    # remove_background(image_path)
    image = preprocess_image(image_path)
    with torch.no_grad():  # 在测试阶段不计算梯度
        output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()

# 主函数
@app.route('/predict', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            # 保存上传的图片
            upload_folder = 'uploads'  # 创建一个文件夹来保存上传的图片
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            image_path = os.path.join(upload_folder, file.filename)
            file.save(image_path)

            # 加载模型
            model = load_model()
            
            # 进行预测
            prediction = predict_image(image_path, model)
            predicted_class = ['freshripe', 'freshunripe', 'overripe', 'ripe', 'rotten', 'unripe'][prediction]
            # 返回预测结果
            return jsonify(f'Predicted class: {predicted_class}')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
