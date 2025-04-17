import numpy as np



class LinearRegressor:
    def linear(self, weights: np.ndarray, bias: np.ndarray, x: np.ndarray):
        return np.dot(weights, x) + bias if bias is not None else np.dot(weights, x)
    def __init__(self,
                 input_size:int,
                 output_size:int,
                 with_bias:bool=True
                 ):
        # 初始化参数
        self.input_size = input_size
        self.output_size = output_size
        self.with_bias = with_bias

        # 初始化权重和偏置
        self.weights = np.random.randn(output_size, input_size)
        if with_bias:
            self.bias = np.random.randn(output_size, 1)
        else:
            self.bias = None

        # 暂存梯度
        self.grad_weights = np.zeros(self.weights.shape)
        if with_bias:
            self.grad_bias = np.zeros(self.bias.shape)
        else:
            self.grad_bias = None

        # 暂存输入
        self.input = None
        # 暂存输出
        self.output = None

    def forward(self, x:np.ndarray):
        self.input = x
        self.output = self.linear(self.weights, self.bias, x)
        return self.output


    def grad(self):
        h=1e-5
        if self.input is None:
            raise ValueError('must carry forward before grading')
        if self.output is None:
            raise ValueError('must carry forward before grading')

        for j in range(self.input_size):
            weight1=self.weights.copy()
            weight1[:,j]+=h
            weight2=self.weights.copy()
            weight2[:,j]-=h

            #print(self.linear(weight1, self.bias, self.input))
            #print(self.grad_weights[:,j])
            self.grad_weights[:,j] += np.squeeze((self.linear(weight1, self.bias, self.input)-self.linear(weight2, self.bias, self.input))/2/h)

        if self.with_bias:
            bias=self.bias.copy()
            bias1=bias+h
            bias2=bias-h
            output1=self.linear(self.weights, bias1, self.input)
            output2=self.linear(self.weights, bias2, self.input)
            self.grad_bias += (output1-output2)/2/h



        #print(f"""对于输入:\n{self.input}\n梯度为:""")
        #print(self.grad_weights)
        #print(self.grad_bias)

    def backward(self, grad_next_layer:np.ndarray , lr:float ,batch_size:int):
        # 链式法则
        grad_bias = (self.grad_bias * grad_next_layer)/batch_size
        grad_weights = self.grad_weights * grad_next_layer/batch_size

        # 调整参数
        self.weights -= lr * grad_weights
        self.bias -= lr * grad_bias

        # 清零梯度
        self.grad_weights = np.zeros(self.weights.shape)
        self.grad_bias = np.zeros(self.bias.shape)


if __name__ == '__main__':
    input_size = 5
    output_size = 2
    with_bias = True

    lr = LinearRegressor(input_size, output_size, with_bias)
    print(lr.weights)
    print(lr.bias)
    for i in range(1000):

        x = np.random.randn(input_size, 1)
        lr.forward(x)
        lr.grad()
        lr.backward(np.random.randn(output_size, 1), lr=0.01, batch_size=1)
    print(lr.weights)
    print(lr.bias)

