# 手撕梯度下降算法

在这一节里面，我们将用一个例子来详细解释梯度下降算法的工作原理，并用Python代码来实现该例子。

## 例子：网络示意图

![gradient_descent.svg](assets/gradient_descent.svg?t=1699850467166)

### 已知条件：

1. 输入：$x_1=1; x_2=2$
2. 初始权重：$w_{11}=1; w_{12}=1; w_{21}=1; w_{22}=-1; w_{a1}=1; w_{a2}=-1$
3. 初始偏置量(Bias)：$b_{11}=1; b_{12}=2; b_{21}=1$
4. 输出：$Y=2$
5. 激活函数(Activate Function): Relu
6. 学习率(Learning rate): $\eta=0.01$
7. 损失函数(Loss Function)：$J(w,b)$: 采用最小二乘法(MSE), 即 $J(w,b) = \frac{1}{2}\cdot (y-Y)^2$

## 解题步骤

### Step 1/3: 前向传播过程

#### 计算$z_1; a_1; z_2; a_2; y$的值

$ z_1 = w_{11}\cdot x_1 + w_{21}\cdot x_2 + b_{11} = 1\times1 + 1\times2 + 1 = 4 $
$ a_1 = Relu(z_1) = 4$
$ z_2 = w_{12}\cdot x_1 + w_{22}\cdot x_2 + b_{12} = 1\times1 + (-1)\times2 + 1 = 0 $
$ a_2 = Relu(z_2) = 0$
$ y = w_{a1}\cdot a_1 + w_{a2}\cdot a_2  + b_{21} = 1\times4 + (-1)\times0 + 1 = 5$

### Step 2/3: 反向传播过程

#### 计算权重$w_{11}$梯度

$w_{11-grad} = \frac{\partial(J)}{\partial(w_{11})} = \frac{\partial(J)}{\partial(y)} \cdot \frac{\partial(y)}{\partial(a_1)}\cdot \frac{\partial(a_1)}{\partial(z_1)}\cdot \frac{\partial(z_1)}{\partial(W_{11})} = (y-Y)\cdot w_{a1} \cdot 1 \cdot x_1 = (5-2)\times1\times1\times1 = 3$

#### 计算$\widehat{w_{11}}$

$\widehat{w_{11}} = w_{11} - \eta \cdot w_{11-grad} = 1 - 0.01 \times 3 = 0.97$
同理，计算其他参数 $\widehat{w_{12}}; \widehat{w_{21}}; \widehat{w_{22}}; \widehat{w_{a1}}; \widehat{w_{a2}}$ 的值

### Step 3/3: 更新模型参数

用新的参数值 $\widehat{w_{11}}; \widehat{w_{12}}; \widehat{w_{21}}; \widehat{w_{22}}; \widehat{w_{a1}}; \widehat{w_{a2}}$ 替换掉原来的参数 $w_{11}; w_{12}; w_{21}; w_{22}; w_{a1}; w_{a2} $，标志着一轮训练已经完成

## 实现代码

```python
import torch
import torch.nn.functional as F

# Define sample tensor
x1 = torch.tensor(1)
x2 = torch.tensor(2)
Y = torch.tensor(2)

# Define weight, bias and learning rate
w11 = torch.tensor(1.0, requires_grad=True)
w12 = torch.tensor(1.0, requires_grad=True)
w21 = torch.tensor(1.0, requires_grad=True)
w22 = torch.tensor(-1.0, requires_grad=True)
wa1 = torch.tensor(1.0, requires_grad=True)
wa2 = torch.tensor(-1.0, requires_grad=True)
b11, b12, b21 = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
learning_rate = 0.01

# Step 1/3: Forward propagation
z1 = w11 * x1 + w12 * x2 + b11
a1 = F.relu(z1)
z2 = w21 * x1 + w22 * x2 + b12
a2 = F.relu(z2)
y = wa1 * a1 + wa2 * a2 + b21

# Calculate loss
loss = ((y - Y)**2)/2

# Step 2/3: Backward propagation
# calculate gradients
w11_grad = (y - Y) * wa1 * 1 * x1
print(w11_grad)

# Step 3/3: Update parameters
w11 = w11 - learning_rate * w11_grad
print(w11)
```

