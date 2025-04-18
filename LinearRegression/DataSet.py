import numpy as np

class DataSet:
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def k_fold(self, k: int, shuffle: bool = True):
        """
        将数据集划分为k折，返回生成器，每次产生(train_set, test_set)

        参数:
            k: 折数
            shuffle: 是否打乱数据顺序
        返回:
            生成器，每次迭代返回一个元组(train_set, test_set)
        """
        if k <= 0:
            raise ValueError("k must be greater than 0")
        if k > len(self.feature):
            raise ValueError("k must be less than or equal to the number of samples")

        # 获取打乱或顺序的索引
        if shuffle:
            indices = np.random.permutation(len(self.feature))
        else:
            indices = np.arange(len(self.feature))

        # 计算每折的大小
        fold_sizes = np.full(k, len(self.feature) // k, dtype=int)
        fold_sizes[:len(self.feature) % k] += 1  # 处理不能整除的情况

        current = 0
        for fold_size in fold_sizes:
            # 获取测试集的索引范围
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]

            # 训练集是除了测试集之外的所有数据
            train_indices = np.concatenate([indices[:start], indices[stop:]])

            # 创建训练集和测试集的DataSet对象
            train_set = DataSet(
                feature=self.feature[train_indices],
                label=self.label[train_indices]
            )
            test_set = DataSet(
                feature=self.feature[test_indices],
                label=self.label[test_indices]
            )

            yield train_set, test_set
            current = stop

    def split(self, test_size=0.2, shuffle=True, random_state=None, stratify=False):
        """
        手动实现数据集按比例分割

        参数:
            test_size: 测试集比例 (0.0 < test_size < 1.0)
            shuffle: 是否打乱数据顺序
            random_state: 随机种子 (保证可复现)
            stratify: 是否按标签分层抽样 (保持类别比例)

        返回:
            train_set, test_set: 两个DataSet对象
        """
        # 参数校验
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if random_state is not None:
            np.random.seed(random_state)

        n_samples = len(self.feature)
        test_num = int(n_samples * test_size)

        # 分层抽样逻辑
        if stratify:
            if self.label is None:
                raise ValueError("stratify requires label data")

            # 获取每个类别的样本索引
            classes, class_indices = np.unique(self.label, return_inverse=True)
            class_counts = np.bincount(class_indices)

            # 计算每个类别在测试集中的数量
            test_counts = (class_counts * test_size).astype(int)
            test_counts = np.maximum(test_counts, 1)  # 确保每类至少有1个测试样本

            test_indices = []
            for cls in range(len(classes)):
                cls_indices = np.where(self.label == classes[cls])[0]
                if shuffle:
                    np.random.shuffle(cls_indices)
                test_indices.extend(cls_indices[:test_counts[cls]])

            if shuffle:
                np.random.shuffle(test_indices)
            test_indices = np.array(test_indices)
        else:
            # 普通随机抽样
            indices = np.arange(n_samples)
            if shuffle:
                np.random.shuffle(indices)
            test_indices = indices[:test_num]

        # 生成训练集索引 (所有不在测试集的样本)
        mask = np.ones(n_samples, dtype=bool)
        mask[test_indices] = False
        train_indices = np.where(mask)[0]

        # 构建分割后的DataSet对象
        train_set = DataSet(
            feature=self.feature[train_indices],
            label=self.label[train_indices] if self.label is not None else None
        )
        test_set = DataSet(
            feature=self.feature[test_indices],
            label=self.label[test_indices] if self.label is not None else None
        )

        return train_set, test_set



if __name__ == "__main__":
    # 示例数据
    features = np.array([[i,i+1] for i in range(10)])  # 10个样本，每个样本1个特征
    labels = np.array([i % 2 for i in range(10)])  # 二分类标签

    # 创建DataSet对象
    dataset = DataSet(features, labels)



    # 进行手动分割
    train_set, test_set = dataset.split(test_size=0.2, shuffle=True, random_state=42, stratify=True)

    print("Train Set:")
    print("Features:")
    print(train_set.feature)
    print("Labels:")
    print(train_set.label)
    print("Test Set:")
    print("Features:")
    print(test_set.feature)
    print("Labels:")
    print(test_set.label)