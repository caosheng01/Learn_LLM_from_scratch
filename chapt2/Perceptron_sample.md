# 手撕梯度下降算法

## 网络示意图

![Perceptron_sample.png](assets/Perceptron_sample.png)

### 已知条件：

1. 输入：$x_1=1; x_2=2$
2. 初始权重：$w_{11}=1; w_{12}=1; w_{21}=1; w_{22}=-1; w_{a1}=1; w_{a2}=-1$
3. 初始偏置量(Bias)：$b_1=1; b_2=2; b_y=1$
4. 输出：$Y=2$
5. 激活函数(Activate Function): Relu
6. 学习率(Learning rate): $\eta=0.01$
7. 损失函数(Loss Function)：$J(w,b)$: 采用最小二乘法(MSE), 即 $J(w,b) = \frac{1}{2}\cdot (y-Y)^2$

## 解题步骤

### Step 1/3: 前向传播过程

#### 计算$z_1; a_1; z_2; a_2; y$的值

$ z_1 = w_{11}\cdot x_1 + w_{21}\cdot x_2 + b_1 = 1\times1 + 1\times2 + 1 = 4 $
$ a_1 = Relu(z_1) = 4$
$ z_2 = w_{12}\cdot x_1 + w_{22}\cdot x_2 + b_2 = 1\times1 + (-1)\times2 + 1 = 0 $
$ a_2 = Relu(z_2) = 0$
$ y = w_{a1}\cdot a_1 + w_{a2}\cdot a_2  + b_y = 1\times4 + (-1)\times0 + 1 = 5$

### Step 2/3: 反向传播过程

#### 计算权重$w_{11}$梯度

$w_{11-grad} = \frac{\partial(J)}{\partial(w_{11})} = \frac{\partial(J)}{\partial(y)} \cdot \frac{\partial(y)}{\partial(a_1)}\cdot \frac{\partial(a_1)}{\partial(z_1)}\cdot \frac{\partial(z_1)}{\partial(W_{11})} = (y-Y)\cdot 1 \cdot 1 \cdot x_1 = (5-2)\times1\times1\times1 = 3$

#### 计算$\widehat{w_{11}}$

$\widehat{w_{11}} = w_{11} - \eta \cdot w_{11-grad} = 1 - 0.01 \times 3 = 0.97$
同理，计算其他参数 $\widehat{w_{12}}; \widehat{w_{21}}; \widehat{w_{22}}; \widehat{w_{a1}}; \widehat{w_{a2}}$ 的值

### Step 3/3: 更新模型参数

用新的参数值 $\widehat{w_{11}}; \widehat{w_{12}}; \widehat{w_{21}}; \widehat{w_{22}}; \widehat{w_{a1}}; \widehat{w_{a2}}$ 替换掉原来的参数 $w_{11}; w_{12}; w_{21}; w_{22}; w_{a1}; w_{a2} $，标志着一轮训练以完成

