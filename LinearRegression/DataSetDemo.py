import numpy as np
from sklearn.datasets import make_classification

# 生成二分类数据集
n_samples = 100  # 样本数
n_features = 2   # 特征维度（二维方便可视化）
X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=2,  # 有效特征数
    n_redundant=0,    # 冗余特征数
    n_classes=2,      # 二分类
    random_state=42   # 随机种子
)

# 确保标签为二维矩阵 (n_samples, 1)
y = y.reshape(-1, 1)

print("Features shape:", X.shape)  # (100, 2)
print("Labels shape:", y.shape)    # (100, 1)
print((type(X), type(y)))