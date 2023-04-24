from PIL import Image
import numpy as np

def dark_channel(image, size):
    """
    计算图像的暗通道
    """
    # 获得图像的红、绿、蓝通道
    r, g, b = image.split()
    # 计算三个通道的最小值
    min_channel = np.minimum(np.minimum(r, g), b)
    # 使用矩形滤波器对最小值图像进行滤波
    kernel = np.ones((size, size), np.uint8)
    dark = cv2.erode(min_channel, kernel)
    # 返回暗通道图像
    return dark

def estimate_transmission(dark, A):
    """
    估计透射率
    """
    transmission = 1 - A*np.float64(dark)
    return transmission

def recover(image, transmission, A, t0=0.1):
    """
    通过估计的透射率恢复图像
    """
    # 对透射率进行裁剪
    transmission = np.maximum(transmission, t0)
    # 计算恢复的图像
    recover = np.zeros_like(image, dtype=np.float64)
    for i in range(3):
        recover[:,:,i] = (image[:,:,i] - A)/transmission + A
    recover = np.uint8(np.minimum(np.maximum(recover, 0), 255))
    return recover

def dehaze(image, size=15, w=0.95, t0=0.1):
    """
    去雾函数
    """
    # 将图像转换为浮点数
    image = np.float64(image)
    # 计算暗通道
    dark = dark_channel(image, size)
    # 估计大气光A
    A = np.max(image, axis=2)
    # 估计透射率
    transmission = estimate_transmission(dark, A)
    # 进行全局大气光A的调整
    A_tilde = np.uint8(np.mean(np.mean(A[np.nonzero(transmission)])))
    A = w*A + (1-w)*A_tilde
    # 恢复图像
    recover = recover(image, transmission, A, t0)
    return Image.fromarray(recover)
