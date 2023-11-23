# Terminology

## 数学部分

### 标量，向量，矩阵和张量的关系

* 标量(scalar): 就是**0维**张量，代码里里用**变量**表示
* 向量(vertor): 就是**1维**张量，代码里里用**一维数组**表示
* 矩阵(matrix): 就是**2维**张量，代码里里用**二维数组**表示

### 几何向量

向量拥有大小和方向。

#### 几何向量的表示法

* 用几何方法来表示向量，如下图这样用箭头来表示的二位向量。
  ![LLMVector.svg](assets/LLM-Vector.svg)
* 用纵向排列方式表示向量，这样的向量被称为列向量。
  
  $\vec{a}=\left[\begin{matrix}3\\1 \end{matrix}\right], \vec{b}=\left[\begin{matrix}2\\3 \end{matrix}\right]$

#### 向量的四则运算

$\vec{a}+\vec{b}=\left[\begin{matrix}3\\1 \end{matrix}\right]+\left[\begin{matrix}2\\3 \end{matrix}\right]=\left[\begin{matrix}3+2\\1+3 \end{matrix}\right]=\left[\begin{matrix}5\\4 \end{matrix}\right]$

$\vec{a}-\vec{b}=\left[\begin{matrix}3\\1 \end{matrix}\right]+\left[\begin{matrix}2\\3 \end{matrix}\right]=\left[\begin{matrix}3-2\\1-3 \end{matrix}\right]=\left[\begin{matrix}1\\-2 \end{matrix}\right]$

$\vec{a} \cdot \vec{b} = a_1b_1+a_2b_2 = 3\times2+1\times3=9$
Tips: 向量**点积**（dot product, 也称**内积**）后得到的已经不是向量了，而是一个**标量**，所以点积也称**标量积**。

#### 余弦定理的向量表示法

我们先介绍几个概念，然后在推导出余弦定理。

* $\Vert a \Vert$表示向量的**长度**（也可以理解为**距离**）。
  
  * 假如二维向量$\vec{a} = (a_1,a_2)$，那么 $\Vert a \Vert = \sqrt{a_1^2 + a_2^2}$
* 假设向量$\vec{a}$和$\vec{b}$之间的夹角为$\theta$, 如何计算$\vec{a} \cdot \vec{b}$呢？
  
  * 先做个方向的转换，我们把$\vec{b}$投影到$\vec{a}$上，这样$\vec{b}$在$\vec{a}$方向上的投影就变成了$\begin{Vmatrix} b \end{Vmatrix} \cos\theta$
  * $\vec{a}$在自己方向上的投影就是$\Vert a \Vert$
  * 这样，$\vec{a} \cdot \vec{b}$ 就等价于 $\Vert a \Vert \Vert b \Vert \cos\theta$，即$  \vec{a} \cdot \vec{b} = \Vert a \Vert \Vert b \Vert \cos\theta  $

简单做一个等式变化，我们就得到了二维向量的余弦定理。

$$
\cos\theta  = \frac{\vec{a} \cdot \vec{b}}{\Vert a \Vert \Vert b \Vert} = \frac{a_1b_1+a_2b_2}{\sqrt{a_1^2 + a_2^2}\sqrt{b_1^2 + b_2^2}}
$$

推广到N维向量空间，就得到公式

$$
\vec{a} = (a_1,a_2,...,a_n);\qquad \vec{b} = (b_1,b_2,...,b_n)
$$

$$
\cos\theta  = \frac{\vec{a} \cdot \vec{b}}{\Vert a \Vert \Vert b \Vert} = \frac{a_1b_1+a_2b_2+...+a_nb_n}{\sqrt{a_1^2+a_2^2+...+a_n^2}\sqrt{b_1^2+ b_2^2+...+b_n^2}}
$$

通常，我们用余弦定理来进行相似度的计算。

* 如果两个向量夹角很小，cos值大于0，接近1，说明他们很相似，即**正相关**。
* 如果两个向量夹角是90度，cos值为0，说明他们不相似，是正交的。
* 如果两个向量夹角大于90度，cos值为小于0，说明他们不相似，即**负相关**。

#### 相似度(Similarity)或距离(distance)

详见《统计学习方法》P255

常用的距离量度

1. 闵可夫斯基距离
   1. 欧氏距离
   2. 曼哈顿距离
   3. 切比雪夫距离
2. 马哈拉诺比斯距离(Mahalanobis distance)
   常用的相似度量度
3. 相关系数(Correlation coefficient)
4. 夹角余弦(Cosine)

---

## 机器学习部分

### 均方误差(MSE:Mean Square Error)

**公式：**$\frac{1}{n}\sum_{i=1}^n(y^{(i)}-f_{\theta}(x^{(i)}))^2$

### Accuracy，Precision和Recall

在分类问题中，我们经常需要计算Accuracy的值来评估模型训练的结果。

| 分类 | 结果标签-True | 结果标签-False |
| --- | --- | --- |
| Positive | True Positive(TP) | False Positive(FP) |
| Negative | False Negative(FN) | True Negative(TN) |

* 正确率(Accuracy)
  
  $$
  Accuracy=\frac{TP+TN}{TP+FP+FN+TN}
  $$
  
  $$
  
  $$
* 精确率(Precision)
  
  $$
  Precision=\frac{TP}{TP+FP}
  $$
* 召回率(Recall)
  
  $$
  Recall=\frac{TP}{TP+FN}
  $$

