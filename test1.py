from PIL import Image
import numpy as np

def dark_channel(image, size):
    # 将RGB图像转换为灰度图像
    if image.mode == 'RGB':
        gray = image.convert('L')
    else:
        gray = image
    # 计算每个像素点的暗通道
    min_gray = np.zeros((gray.height, gray.width))
    for i in range(gray.height):
        for j in range(gray.width):
            min_gray[i,j] = np.min(gray.crop((j-size//2, i-size//2, j+size//2, i+size//2)))
    # 对暗通道进行一个平滑处理
    kernel = np.ones((size, size)) / (size * size)
    kernel = kernel.astype(float)
    min_gray = min_gray.astype(float)
    if min_gray.ndim == 3:
        dark_smooth = np.zeros((gray.height, gray.width, 3))
        for k in range(3):
            dark_smooth[:, :, k] = np.convolve(min_gray[:, :, k], kernel, mode='same')
    else:
        dark_smooth = np.convolve(min_gray, kernel, mode='same')
    # 返回平滑后的暗通道图像
    return dark_smooth



def estimate_transmission(image, size):
    # 计算暗通道图像
    dark = dark_channel(image, size)
    # 估计雾浓度
    transmission = 1 - 0.95 * dark / 255
    # 返回估计出的雾浓度
    return transmission

def dehaze(image, size):
    # 估计雾浓度
    transmission = estimate_transmission(image, size)
    # 对原始图像进行去雾处理
    dehazed = np.zeros_like(image)
    for k in range(3):
        dehazed[:, :, k] = (image[:, :, k] - transmission * 255) / np.maximum(transmission, 0.1)
    # 返回去雾后的图像
    return Image.fromarray(np.uint8(np.clip(dehazed, 0, 255)))

if __name__ == '__main__':
    # 输入图像的长宽像素数目
    width = 440
    height = 660
    # 读取输入图像
    image = Image.open('input.jpg').resize((width, height))
    # 雾霾去除的窗口大小
    size = 15
    # 进行去雾处理
    dehazed = dehaze(image, size)
    # 显示去雾后的图像
    dehazed.show()
