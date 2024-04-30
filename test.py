import torch
import torchvision
from torchvision import transforms
from PIL import Image
import pandas as pd
import random
import os

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
    model.load_state_dict(torch.load('model_epoch_15.pth', map_location=device))
    
    model.eval()  # 设置为评估模式
    return model

# 预测函数
def predict_image(image_path, model):
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

