import numpy as np
# 定义样本数量
n_samples = 500  # 样本数
true_weights = np.array([[1, 1]])  # 真实权重 [w1=5, w2=2]
true_bias = 1                   # 真实偏置

# 生成输入 X (形状: 2 x n_samples，每列为一个样本)
np.random.seed(42)  # 固定随机种子
X = np.random.randn(2, n_samples)  # x1和x2服从标准正态分布

# 计算 y (形状: 1 x n_samples)
y = true_weights @ X + true_bias  # 矩阵乘法实现 y = 5*x1 + 2*x2 + 3

# 添加高斯噪声（可选，模拟真实数据）
noise = np.random.randn(1, n_samples) * 0.1  # 噪声标准差=0.1
y_noisy = y + noise

# 打印前5个样本
print("输入 X (前5个样本, 每列为一个样本):\n", X[:, :5])
print("\n输出 y (前5个样本):\n", y_noisy[:, :5])

from LossFunction import LossFunction
from LinearRegressor import LinearRegressor
from Optimizer import Optimizer

if __name__ == '__main__':
    model = LinearRegressor(input_size=2, output_size=1, with_bias=True)
    loss_func = LossFunction('mse')
    optimizer = Optimizer(model, loss_func, learning_rate=0.01)
    optimizer.train(X, y_noisy)
    print("训练结束：")
    print(f"w\n{model.weights}")
    print(f"b\n{model.bias}")