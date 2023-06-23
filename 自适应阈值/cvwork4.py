import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def otsu_threshold(image):
    # 计算图像直方图
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    # 归一化直方图
    hist_norm = hist.ravel() / hist.max()
    
    # 初始化类间方差和最佳阈值
    best_threshold = 0
    max_variance = 0
    
    for threshold in range(256):
        # 计算类内和类间方差
        w0 = np.sum(hist_norm[:threshold])
        w1 = np.sum(hist_norm[threshold:])
        u0 = np.sum(np.arange(threshold) * hist_norm[:threshold]) / (w0 + 1e-6)
        u1 = np.sum(np.arange(threshold, 256) * hist_norm[threshold:]) / (w1 + 1e-6)
        variance = w0 * w1 * (u0 - u1) ** 2
        
        # 更新最佳阈值和类间方差
        if variance > max_variance:
            max_variance = variance
            best_threshold = threshold
    
    # 应用最佳阈值进行二值化
    _, thresholded = cv2.threshold(image, best_threshold, 255, cv2.THRESH_BINARY)
    
    return thresholded

def iterative_threshold(image, epsilon=1e-6):
    # 初始化阈值
    threshold = image.mean()
    
    while True:
        # 分割图像
        below_threshold = image < threshold
        above_threshold = image >= threshold
        
        # 计算均值
        mean_below = np.mean(image[below_threshold])
        mean_above = np.mean(image[above_threshold])
        
        # 更新阈值
        new_threshold = 0.5 * (mean_below + mean_above)
        
        # 判断迭代是否收敛
        if np.abs(threshold - new_threshold) < epsilon:
            break
        
        threshold = new_threshold
    
    # 应用最佳阈值进行二值化
    _, thresholded = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    return thresholded

if __name__ == '__main__':
    # 读取图像
    image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
    
    # 大津法阈值分割
    otsu_result = otsu_threshold(image)
    
    # 迭代法阈值分割
    iterative_result = iterative_threshold(image)
    
    # 比较两种方法的结果
    diff = cv2.absdiff(otsu_result, iterative_result)


    
    # 保存结果图像
    cv2.imwrite('otsu_result.jpg', otsu_result)
    cv2.imwrite('iterative_result.jpg', iterative_result)
    cv2.imwrite('differences.jpg', diff)


    # 计算图像的 PSNR 和 SSIM 值
    psnr_otsu = peak_signal_noise_ratio(image, otsu_result)
    ssim_otsu = structural_similarity(image, otsu_result)
    
    psnr_iterative = peak_signal_noise_ratio(image, iterative_result)
    ssim_iterative = structural_similarity(image, iterative_result)
    
    # 打印结果
    print("大津法阈值分割的 PSNR 值:", psnr_otsu)
    print("大津法阈值分割的 SSIM 值:", ssim_otsu)
    print("迭代法阈值分割的 PSNR 值:", psnr_iterative)
    print("迭代法阈值分割的 SSIM 值:", ssim_iterative)
