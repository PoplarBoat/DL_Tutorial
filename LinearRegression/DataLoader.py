import numpy as np
from DataSet import DataSet


class DataLoader:
    def __init__(self,dataset: DataSet, batch_size: int):
        self.dataset=dataset
        self.batch_size=batch_size

    def get_batch(self):
        for i in range(0,len(self.dataset.feature),self.batch_size):
            yield self.dataset.feature[i:i+self.batch_size],self.dataset.label[i:i+self.batch_size]


if __name__=='__main__':
    features = np.array([[i,i+1] for i in range(10)])
    labels = np.array([i % 2 for i in range(10)])
    dataset = DataSet(features, labels)
    dataloader = DataLoader(dataset, batch_size=3)
    for feature, label in dataloader.get_batch():
        print(feature)
        print(label)