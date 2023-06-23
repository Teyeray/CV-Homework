import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def dpcm_encode(image, quantization_bits):
    height, width = image.shape
    
    # 预测误差矩阵
    errors = np.zeros((height, width), dtype=np.int16)
    
    # 量化器
    quantizer_levels = 2 ** quantization_bits - 1
    
    # DPCM编码
    for y in range(1, height):
        for x in range(1, width):
            predicted_value = image[y-1, x-1]
            prediction_error = int(image[y, x]) - predicted_value
            quantized_error = np.round(prediction_error / quantizer_levels) * quantizer_levels
            errors[y, x] = int(quantized_error)
    
    return errors

def dpcm_decode(errors, quantization_bits):
    height, width = errors.shape
    
    # 量化器
    quantizer_levels = 2 ** quantization_bits - 1
    
    # DPCM解码
    decoded_image = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(1, height):
        for x in range(1, width):
            predicted_value = decoded_image[y-1, x-1]
            quantized_error = errors[y, x]
            prediction_error = quantized_error * quantizer_levels
            pixel_value = predicted_value + prediction_error
            decoded_image[y, x] = np.clip(pixel_value, 0, 255)
    
    return decoded_image

def calculate_psnr(original_image, reconstructed_image):
    mse = np.mean((original_image - reconstructed_image) ** 2)
    psnr = 10 * np.log10((255 ** 2) / mse)
    return psnr

def calculate_ssim(original_image, reconstructed_image):
    ssim_value = ssim(original_image, reconstructed_image, data_range=reconstructed_image.max() - reconstructed_image.min())
    return ssim_value

if __name__ == '__main__':
    # 读取灰度图像
    image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
    
    # 编码和解码比较
    quantization_bits = [1, 2, 4, 8]
    
    for bits in quantization_bits:
        # DPCM编码
        encoded_errors = dpcm_encode(image, bits)
        
        # DPCM解码
        reconstructed_image = dpcm_decode(encoded_errors, bits)
        
        # 计算PSNR和SSIM值
        psnr = calculate_psnr(image, reconstructed_image)
        ssim_value = calculate_ssim(image, reconstructed_image)
        
        print(f'Quantization bits: {bits}')
        print(f'PSNR: {psnr:.2f} dB')
        print(f'SSIM: {ssim_value:.4f}\n')
