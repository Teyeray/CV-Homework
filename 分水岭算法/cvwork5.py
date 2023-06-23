import cv2
import numpy as np

def watershed_segmentation(input_image_path, output_image_path):
    # 读取输入图像
    input_image = cv2.imread(input_image_path)
    
    # 转换为灰度图像
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    # 对图像进行阈值处理，以获得二值图像
    ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # 对图像进行形态学操作，以去除噪音和不规则区域
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # 寻找未知区域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # 标记分水岭区域
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # 应用分水岭算法
    markers = cv2.watershed(input_image, markers)
    input_image[markers == -1] = [0, 0, 255]
    
    # 保存输出图像
    cv2.imwrite(output_image_path, input_image)


if __name__ == '__main__':
    input_image_path = 'input_image.jpg'  # 输入图像路径
    output_image_path = 'output_image.jpg'  # 输出图像路径
    
    watershed_segmentation(input_image_path, output_image_path)
