import numpy as np
from mpmath import sigmoid


class Linear:
    def __init__(self,input_size:int,output_size:int,with_bias:bool=True):
        self.input_size=input_size
        self.output_size=output_size
        self.with_bias=with_bias

        # 初始化权重，列向量为一个输出节点关于输入的权重向量
        # 这样，x @ weights 可以得到行向量的组合，每一个行向量是输出向量
        self.weights=np.random.randn(input_size,output_size)

        # 初始化偏置项，大小和输出向量一致
        if with_bias:
            self.bias=np.random.randn(1,output_size)
        else:
            self.bias=None

        # 暂存输入输出值
        self.input=None
        self.output=None

        # 暂存梯度
        self.grad_w=np.zeros_like(self.weights)
        self.grad_b=np.zeros_like(self.bias) if self.with_bias else None

    def __call__(self,x:np.array):
        self.input = x
        self.output = x @ self.weights + self.bias if self.with_bias else x @ self.weights
        return x @ self.weights + self.bias if self.with_bias else x @ self.weights

    def backward(self, grad_next: np.array):
        h = 1e-6
        if grad_next is None:
            grad_next = np.ones((1, self.output_size))
        # 由于weights矩阵中，每一个列向量对应一个输出节点，而每一个行向量的元素都分属于不同输出节点，所以可以一次计算一行的梯度
        # 中心差分法
        for i in range(self.weights.shape[0]):
            x = self.input
            w = self.weights.copy()
            w[i] += h
            o1 = x @ w + self.bias if self.with_bias else x @ w
            w[i] -= 2*h
            o2 = x @ w + self.bias if self.with_bias else x @ w

            grad=(o1 - o2) / (2 * h)
            self.grad_w[i] += np.squeeze((o1 - o2) / (2 * h))
        self.grad_w *= grad_next
        #print(self.grad_w)
        if self.with_bias:
            b = self.bias.copy()
            b += h
            o1 = x @ self.weights + b
            b -= 2*h
            o2 = x @ self.weights + b
            self.grad_b += (o1 - o2) / (2 * h)
            print(self.grad_b)
            self.grad_b *= grad_next

class Sigmoid:
    def __init__(self,input_size:int):
        self.input_size=input_size
        self.output_size=input_size

        # 暂存梯度
        self.grad_input=np.zeros((1,self.input_size))
        # 暂存输入输出
        self.input=None
        self.output=None


    def __call__(self,x:np.array):
        self.input=x
        self.output=1/(1+np.exp(-x))
        return self.output

    def backward(self,grad_next:np.array):
        if grad_next is None:
            grad_next=np.ones((1,self.output_size))
        self.grad_input=self.output*(1-self.output)*grad_next


class MSE:
    def __init__(self):
        # 暂存梯度
        self.grad_y_pred=np.zeros((1,1))
        # 暂存输入输出
        self.y_pred=None
        self.y_true=None
        self.output=None

    def __call__(self,y_true:np.array,y_pred:np.array):
        self.y_pred=y_pred
        self.y_true=y_true
        self.output=np.mean((y_true-y_pred)**2)
        return self.output

    def backward(self,grad_next:np.array):
        if grad_next is None:
            grad_next=np.ones((1,1))
        self.grad_y_pred=2*(self.y_true-self.y_pred)*grad_next


class CategoricalCrossEntropy:
    def __init__(self,num_classes:int):
        self.num_classes=num_classes

        # 暂存梯度
        self.grad_y_pred=np.zeros((1,num_classes))

        # 暂存输入输出
        self.y_pred=None
        self.y_true=None

    def __call__(self,y_true:np.array,y_pred:np.array):
        self.y_pred=y_pred
        self.y_true=y_true

        self.output=-np.sum(y_true*np.log(y_pred+1e-15))/self.num_classes

        return self.output

    def backward(self,grad_next:np.array):
        if grad_next is None:
            grad_next=np.ones((1,self.num_classes))
        self.grad_y_pred=1/np.log(self.y_pred+1e-15)/self.num_classes




if __name__=="__main__":
    linear=Linear(2,1)
    print(f"w:\n{linear.weights}")
    print(f"b:\n{linear.bias}")

    x=np.array([[3,2]])
    print(f"x:\n{x}")

    linear(x)
    print(f"y:\n{linear.output}")

    sigmoid=Sigmoid(1)
    sigmoid(linear.output)
    print(f"sigmoid(y):\n{sigmoid.output}")

    mse = MSE()
    mse(sigmoid.output, np.array([[0.5]]))
    print(f"mse:\n{mse.output}")

    mse.backward(None)
    print(f"dmse/ds:\n{mse.grad_y_pred}")

    sigmoid.backward(mse.grad_y_pred)
    print(f"dmse/dy:\n{sigmoid.grad_input}")

    linear.backward(sigmoid.grad_input)
    print(f"dmse/dw:\n{linear.grad_w}")




