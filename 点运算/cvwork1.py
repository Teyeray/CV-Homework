from PIL import Image

# 定义线性、分段线性和非线性点运算函数

## 线性
def linear(pixel):
    return (pixel * 1.2 + 30)

##分段线性
def linear2(pixel):
    if pixel < 128:
        return pixel * 2
    else:
        return pixel

##非线性
def nonlinear(pixel):
    return int(pixel ** 0.5) * 16

# 读取灰度图片
img = Image.open('test_gray.jpg').convert('L')

# 线性点运算
linear_img = img.point(linear)
linear_img.save('output_linear_img.jpg')

# 分段线性点运算
piecewise_linear_img = img.point(linear2)
piecewise_linear_img.save('output_piecewise_linear_img.jpg')

# 非线性点运算
nonlinear_img = img.point(nonlinear)
nonlinear_img.save('output_nonlinear_img.jpg')