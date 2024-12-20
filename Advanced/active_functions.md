# 激活函数

## 常用的激活函数

在机器学习（ML）中，激活函数是神经网络中非常重要的一部分，它们用于在神经元之间引入非线性特性，从而使网络能够学习和模拟复杂的非线性关系。以下是一些常见的激活函数及其适用场景：

### 1. Sigmoid函数

* ​**函数表达式**​：**f**(**x**)**=**1**+**e**−**x**1**​
* ​**适用场景**​：Sigmoid函数常用于二分类问题的输出层，因为它可以将任意实值压缩到(0,1)区间内，表示概率。然而，由于其梯度消失问题（当输入值非常大或非常小时，梯度接近0），它并不适合在深度神经网络的隐藏层中使用。

### 2. 双曲正切（Tanh）函数

* ​**函数表达式**​：**f**(**x**)**=**e**x**+**e**−**x**e**x**−**e**−**x**​
* ​**适用场景**​：Tanh函数是Sigmoid函数的改进版，它将输出值压缩到(-1,1)区间内，并且以0为中心。这使得Tanh函数在某些情况下（如输入数据以0为中心时）比Sigmoid函数表现更好。然而，它同样存在梯度消失问题，因此也不适合在深度神经网络的隐藏层中广泛使用。

### 3. 线性整流单元（ReLU）

* ​**函数表达式**​：**f**(**x**)**=**max**(**0**,**x**)**
* ​**适用场景**​：ReLU函数是目前深度学习中最常用的激活函数之一。它对于正输入值保持原样，对于负输入值则输出0。这种特性使得ReLU函数在训练过程中能够更快地收敛，并且在一定程度上缓解了梯度消失问题。ReLU函数还促进了神经网络的稀疏性，即许多神经元在训练过程中会变为0，这有助于提升模型的泛化能力。然而，ReLU函数也存在“死亡神经元”问题，即某些神经元在训练过程中可能永远不会被激活。

### 4. 泄漏线性整流单元（Leaky ReLU）

* ​**函数表达式**​：**f**(**x**)**=**max**(**αx**,**x**)**，其中**α**是一个很小的常数（如0.01）
* ​**适用场景**​：Leaky ReLU是ReLU的一个变体，它解决了ReLU函数中的“死亡神经元”问题。通过允许负输入值有一个小的梯度（由**α**控制），Leaky ReLU使得神经元在训练过程中更有可能被激活。这使得Leaky ReLU在某些情况下比ReLU表现更好。

### 5. 参数化线性整流单元（PReLU）

* ​**函数表达式**​：与Leaky ReLU类似，但**α**是一个可学习的参数
* ​**适用场景**​：PReLU进一步扩展了Leaky ReLU的概念，允许**α**在训练过程中被学习。这使得PReLU能够自适应地调整其行为，以适应不同的数据集和任务。因此，PReLU在某些情况下可能比Leaky ReLU表现更好。

### 6. 指数线性单元（ELU）

* ​**函数表达式**​：**f**(**x**)**=**{**x**α**(**e**x**−**1**)**​**if **x**>**0**if **x**≤**0**​
* ​**适用场景**​：ELU函数结合了ReLU和Sigmoid函数的优点，它在正输入值上保持线性，类似于ReLU；在负输入值上则有一个平滑的过渡，类似于Sigmoid函数。这使得ELU函数能够缓解梯度消失问题，并且在一定程度上减少了神经元的死亡。然而，ELU函数的计算复杂度略高于ReLU和Leaky ReLU。

### 7. 高斯误差线性单元（GELU）

* ​**函数表达式**​：**f**(**x**)**=**x**⋅**Φ**(**x**)**，其中**Φ**(**x**)是高斯函数
* ​**适用场景**​：GELU函数是一种非线性激活函数，它结合了ReLU和Dropout的优点。GELU函数对于正输入值有一个平滑的过渡，类似于ReLU；同时，它还能够通过调整输入值的分布来减少过拟合的风险。这使得GELU函数在某些深度学习模型中表现良好。

### 8. Softmax函数

* ​**函数表达式**​：**f**(**x**i**​**)**=**∑**j**​**e**x**j**​**e**x**i**​**​**
* ​**适用场景**​：Softmax函数通常用于多分类问题的输出层，它将神经网络的输出转换为概率分布。Softmax函数将每个输出值转换为正数，并且所有输出值的和为1，这使得Softmax函数非常适合于表示概率分布。

需要注意的是，以上列举的激活函数并不是全部，实际上还有很多其他的激活函数（如Maxout、Swish等）也被用于不同的机器学习场景中。在选择激活函数时，需要根据具体

