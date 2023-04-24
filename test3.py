import cv2
import numpy as np

def dark_channel(img, size):
    """
    计算暗通道图像
    :param img: 输入图像
    :param size: 窗口大小
    :return: 暗通道图像
    """
    b, g, r = cv2.split(img)
    min_img = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dc_img = cv2.erode(min_img, kernel)
    return dc_img

def get_atmosphere(img, dc_img, p):
    """
    计算全局大气光照
    :param img: 输入图像
    :param dc_img: 暗通道图像
    :param p: 暗通道系数
    :return: 大气光照值
    """
    h, w = img.shape[:2]
    num_pixel = h * w
    num_sample = int(num_pixel * p / 100)
    dc_vec = dc_img.reshape(num_pixel)
    indices = np.argsort(dc_vec)[-num_sample:]
    b, g, r = cv2.split(img)
    max_channel = np.zeros(3)
    for i in range(num_sample):
        x = indices[i] % w
        y = int(indices[i] / w)
        if max_channel[0] < b[y, x]:
            max_channel[0] = b[y, x]
        if max_channel[1] < g[y, x]:
            max_channel[1] = g[y, x]
        if max_channel[2] < r[y, x]:
            max_channel[2] = r[y, x]
    A = np.max(max_channel)
    return A

def dehaze(img, omega=0.95, p=0.1, t0=0.1, size=15):
    """
    图像去雾
    :param img: 输入图像
    :param omega: 调整大气光照的强度
    :param p: 暗通道系数
    :param t0: 最小透射率
    :param size: 暗通道窗口大小
    :return: 去雾图像
    """
    dc_img = dark_channel(img, size)
    A = get_atmosphere(img, dc_img, p)
    t_est = 1 - omega * dc_img / A
    t_est[t_est < t0] = t0
    b, g, r = cv2.split(img)
    img = cv2.merge([cv2.divide((b - A), t_est) + A,
                     cv2.divide((g - A), t_est) + A,
                     cv2.divide((r - A), t_est) + A])
    img[img > 255] = 255
    img[img < 0] = 0
    return np.uint8(img)

# 测试代码
if __name__ == '__main__':
    img = cv2.imread('input.jpg')
    img = cv2.resize(img, (800, 600)) # 调整输入图
    img = cv2.imread('input.jpg')
    img = cv2.resize(img, (800, 600)) # 调整输入图
    result = dehaze(img, omega=0.95, p=0.1, t0=0.1, size=15)
    cv2.imshow('Input Image', img)
    cv2.imshow('Dehazed Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

