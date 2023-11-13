# Terminology

## 数学

### 标量，向量，矩阵和张量

* 标量(scalar): 就是**0维**张量
* 向量(vertor): 就是**1维**张量
* 矩阵(matrix): 就是**2维**张量

### 几何向量

向量拥有大小和方向。

* 用几何方法来表示向量，如下图这样用箭头来表示的二位向量。
  ![LLMVector.svg](assets/LLM-Vector.svg)
* 用纵向排列方式表示向量，这样的向量被称为列向量。
  $\vec{a}=\left[\begin{matrix}1\\2 \end{matrix}\right], \vec{b}=\left[\begin{matrix}2\\3 \end{matrix}\right]$

---

## 机器学习

### 均方误差(MSE:Mean Square Error)

**公式：**$\frac{1}{n}\sum_{i=1}^n(y^{(i)}-f_{\theta}(x^{(i)}))^2$

### Accuracy，Precision和Recall

在分类问题中，我们经常需要计算Accuracy的值来评估模型训练的结果。

| 分类 | 结果标签-True | 结果标签-False |
| --- | --- | --- |
| Positive | True Positive(TP) | False Positive(FP) |
| Negative | False Negative(FN) | True Negative(TN) |

* 正确率(Accuracy)
  **公式：**$Accuracy=\frac{TP+TN}{TP+FP+FN+TN}$
* 精确率(Precision)
  ***公式：**$Precision=\frac{TP}{TP+FP}$
* 召回率(Recall)
  **公式：**$Recall=\frac{TP}{TP+FN}$

