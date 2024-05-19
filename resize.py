import cv2
import numpy as np

def find_largest_non_black_region(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image")
        return
    
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
        zoomscale = longest_side / max(h, w)
        
        # 根据边界框裁剪非黑区域
        cropped = image[y:y+h, x:x+w]
        
        # 等比例放大裁剪的非黑区域
        resized_cropped = cv2.resize(cropped, (int(w*zoomscale), int(h*zoomscale)), interpolation=cv2.INTER_CUBIC)
        
        # 因为只考虑了单个最大的非黑区域，这里直接显示或保存处理后的非黑区域
        cv2.imshow('Resized Non-Black Region', resized_cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No significant non-black region found.")

# 使用函数，传入你的图片路径
find_largest_non_black_region('valid/musa-acuminata-banana-8773b90d-394a-11ec-8949-d8c4975e38aa_jpg.rf.003140273bf797576bd1d4a630793969.jpg')