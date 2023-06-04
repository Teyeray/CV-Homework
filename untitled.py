import cv2
import numpy as np

# 读取原始图像
img = cv2.imread('test.jpg')

# 将图像转换为灰度图像
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 定义直线函数
line_func = np.polyfit([0, 1], [0, 1], deg=1)[0]

# 计算直线方程
x1, y1 = line_func * (-1), line_func * 1
x2, y2 = x1 + 1, y1 + 1

# 绘制直线
cv2.line(gray_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=2)

# 显示结果图像
cv2.imshow('Result', gray_img)
cv2.waitKey(0)
#cv2.destroyAllWindows()
