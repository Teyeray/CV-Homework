from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def main():
    # 读取图像
    image = Image.open('input_image.jpg').convert('L')

    # 转换为NumPy数组
    image_array = np.array(image, dtype=np.float32)

    # 添加模拟的运动模糊
    kernel_size = 15  # 设置核的大小
    angle = 30  # 设置运动模糊的角度
    blurred_array = add_motion_blur(image_array, kernel_size, angle)

    # 添加高斯噪声
    noisy_array = add_gaussian_noise(blurred_array, mean=0, std=10)

    # 维纳滤波
    restored_array = wiener_filter(noisy_array, kernel_size, K=0.01)

    # 将像素值限制在0到255范围内
    restored_array = np.clip(restored_array, 0, 255)

    # 转换为PIL图像
    restored_image = Image.fromarray(restored_array.astype(np.uint8))

    # 显示原始图像和恢复后的图像
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(restored_image, cmap='gray')
    axs[1].set_title('Restored Image')
    axs[1].axis('off')
    plt.show()


def add_motion_blur(image_array, kernel_size, angle):
    # 创建运动模糊核
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    slope = np.tan(np.deg2rad(angle))

    for i in range(kernel_size):
        offset = int(round(slope * (i - center)))
        kernel[i, center + offset] = 1 / kernel_size

    # 对图像进行卷积操作
    blurred_array = np.convolve(image_array.flatten(), kernel.flatten(), mode='same')
    blurred_array = blurred_array.reshape(image_array.shape)

    return blurred_array


def add_gaussian_noise(image_array, mean, std):
    noise = np.random.normal(mean, std, image_array.shape)
    noisy_array = image_array + noise
    return noisy_array


def wiener_filter(image_array, kernel_size, K):
    # 创建维纳滤波器
    psf = np.ones_like(image_array) / kernel_size ** 2

    # 使用傅里叶变换对图像和PSF进行频域计算
    image_fft = np.fft.fft2(image_array)
    psf_fft = np.fft.fft2(psf)

    # 计算维纳滤波器
    restored_fft = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + K)

    # 使用傅里叶逆变换将结果转换回空域
    restored_array = np.fft.ifft2(image_fft * restored_fft)

    # 返回实部作为恢复后的图像
    return np.real(restored_array)


if __name__ == '__main__':
    main()
