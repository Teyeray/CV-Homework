import cv2
import numpy as np

def wiener_filter(image, kernel, K):
    # 使用傅里叶变换将图像和卷积核转换为频域
    image_freq = np.fft.fft2(image)
    kernel_freq = np.fft.fft2(kernel, image.shape)
    
    # 计算卷积结果的幅度谱
    kernel_freq_abs = np.abs(kernel_freq)
    kernel_freq_abs_squared = np.square(kernel_freq_abs)
    
    # 应用维纳滤波器公式
    result_freq = np.conj(kernel_freq) * image_freq / (np.square(kernel_freq) + K)
    
    # 将频域结果转换为空域图像
    result = np.fft.ifft2(result_freq)
    result = np.abs(result)
    result = np.uint8(result)
    
    return result

def calculate_psnr(original, filtered):
    mse = np.mean((original - filtered) ** 2)
    max_pixel = np.max(original)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

if __name__ == '__main__':
    # 读取输入图像
    image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
    
    # 定义模糊核
    kernel = np.ones((3, 3), dtype=np.float32) / 25.0
    
    # 维纳滤波参数
    K = 0.24
    #慢慢试出来的。。
    
    # 应用维纳滤波
    filtered_image = wiener_filter(image, kernel, K)
    
    
    # 保存结果图像
    cv2.imwrite('output_image.jpg', filtered_image)
    
    # 显示原始图像和滤波结果
    cv2.imshow("Original Image", image)
    cv2.imshow("Filtered Image", filtered_image)
    cv2.waitKey(0)
