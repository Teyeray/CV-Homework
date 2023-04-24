import numpy as np
import matplotlib.pyplot as plt

# 数据集
X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2], [4, 3]])
y = np.array([1, 1, 1, -1, -1, -1])

# 计算超平面参数
w = np.zeros(X.shape[1])
b = 0
learning_rate = 0.1
epochs = 1000

for epoch in range(epochs):
    for i, x in enumerate(X):
        if y[i] * (np.dot(w, x) + b) <= 0:
            w = w + learning_rate * y[i] * x
            b = b + learning_rate * y[i]

# 可视化结果
print(w.shape,b.shape)
plt.scatter(X[:, 0], X[:, 1], c=y)
xx, yy = np.meshgrid(np.arange(X[:, 0].min()-1, X[:, 0].max()+1, 0.02), np.arange(X[:, 1].min()-1, X[:, 1].max()+1, 0.02))
Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b).reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.5)
plt.show()