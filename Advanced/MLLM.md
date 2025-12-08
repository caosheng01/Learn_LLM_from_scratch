# 多模态大语言模型（Multimodal LLM）技术简介：聚焦视觉与语言处理。

## 前言：CNN还是Transformer？

回顾AI的发展史，从某种意义上说就是计算机对人脑的仿真，和人类进化史一样，人类先学会从视觉里学会认识事物，先诞生了绘画，然后是语言，最后是文字。现代AI技术的发展的历史轨迹也和人类进化史类似，先解决机器视觉（Visual）的问题，然后再解决自然语言（Language）处理。
新的问题来了，众所周知，解决Visual中问题，依赖于CNN。而解决NLP问题呢，都采用Transformer。从生物仿真学的角度来说，人脑解决视觉和语言问题，有且仅有一个器官。用科研的角度来说，当前AI用CNN来处理图像，用Transformer处理文本的解决方案，不够优美（elgant），甚至有点丑陋（Urgly）。其实，物理学在18世纪后叶也有类似的情况，传统力学用牛顿的一套公式，新兴的电学和磁学用的是另一套公式，最终麦克斯韦统一了这两套系统。自然而然，搞AI的这帮聪明脑袋中，就有很多人开始尝试用统一的方式去处理Visual和Language。

### 本章约定

在展开介绍这帮聪明脑袋的成果之前，先解释一下Visual和Language术语。在处理机器视觉（Visual）问题的时候，主要围绕这处理图片（Image）和视频（Video）这两方面工作。本文的重点是聚焦2D的图像（Image），原因也很简单，当前时间点（2025年底），AI处理Image问题，解决方案相对比较成熟。同时，处理的Video的问题也必须要深入理解如何处理Image的。回到传统的Language定义，包含声音（Voice）和文字（Text）两方面工作，LLM处理语音这类问题时，基本上先把语音转化为文本（ASR过程），LLM处理完，输出文本，最后转成语音（TTS过程）。所以，本文提到Language时，如果没有特别说明，就是指处理文本(Text)的过程。提到Visual时，不特殊说明都是二维图片（Image）处理。

## 多模态的技术基石

上文提到，很多人开始尝试用统一的方法处理Visual和Language。这里我们介绍一个非常有影响力的工作——ViT。

### ViT: 用Transformer统一处理VL问题。

ViT是2020年Google团队提出的将Transformer应用在图像分类的模型，虽然不是第一篇将Transformer应用在视觉任务的论文，但是因为其模型简单，高效且可扩展性强，成为了Transformer在CV领域应用的里程碑著作。其核心思路就是抛弃CNN，将2D图片处理成Token，送给Transformer模型来处理。这样，VL两类完全不同的数据就能用同一个Transformer模型。而实验结果表明，当拥有足够多的数据进行预训练的时候，ViT的表现就会超过CNN，突破Transformer缺少归纳偏置的限制，可以在下游任务中获得较好的迁移效果。

接下来，我们一起来看一下ViT是如何工作的。

#### ViT架构

我们先回顾一下Transformer结构，先把文本（Text）做Word Embedding，转化成Token，然后加上位置编码（Position Embedding）输入到MHA（Multi-Head Attention)。问题来了，如何把二维的图片转化成一维的Token呢？其实这个问题在传统的计算机图形学里有现成的解决方案。我们按照BMP图片格式为例，100x100的图片大小。因为宽度和高度都是100像素，加上RGB三个通道，这个BMP格式的图片存储在计算机磁盘里，就转化成一维的方式了，即3（通道）x100（高度）x100（宽度）。
ViT也是借助这个思路，只是用像素点的分割力度太细了，改用16x16的小图片（Patch）作为最小单元。下图是ViT模型的总览图，图中Patch Embeddding部分就是描述了这个过程。

![MLLM_ViT.png](../images/MLLM_ViT.png)

#### 图像分块（Patch Embedding）

举个例子详细说一下Patch Embedding过程。输入图片大小为224x224，将图片分为固定大小的patch，patch大小为16x16，则每张图像会生成224x224/16x16=196个patch，即输入序列长度为​196​，每个patch维度3x16x16=​768​，线性投射层的维度为768xN (N=768)，因此输入通过线性投射层之后的维度依然为196x768，即一共有196个token，每个token的维度是768。这里还需要加上一个特殊字符CLS，因此最终的Tokens数是197。下图详细展示了这个Patch Embedding的过程。

![MLLM_ViT_Patch.svg](../images/MLLM_ViT_Patch.svg)

#### 位置编码（Position Embedding）

为了保留图像块的空间位置信息，ViT为每个图像块添加一个位置编码。位置编码是一个与Patch Embedding维度相同的向量，表示每个图像块的位置信息。而位置编码可以是可学习的参数，也可以是固定的编码。

* 可学习的位置编码: 直接初始化为可训练的参数，随着模型训练进行优化。
* 固定的位置编码: 使用正弦和余弦函数生成（类似于原始Transformer中的编码方式）。

Tips：在 ViT 的原始实现中，位置编码是可学习的。

最后，将位置编码与Patch Embedding相加，形成最终的输入序列。公式表示为：

$$
Z_0 = [E_{cls}; E_1 + P_1; E_2 + P_2; \dots; E_N + P_N] \\
$$

$$
E_{cls}: 特殊的分类标记[CLS]的Token。\\
E_i: 第 i 个图像块的嵌入。\\
P_i: 第i个图像块的位置编码。
$$

ViT后续的处理，和标准Transformer Encoder模型基本上是一致的。因为本文要讨论的内容比较多，关于ViT，只会介绍最核心的思想，如果读者想了解细节，建议直接读一下原始论文。

### VL模型迎来爆发期

2020年，伴随着ViT的横空出世，大家已经认识到多模态解决方案的大爆发时刻即将来临。短短的2，3年，多模态在VL这个赛道就卷出了结果。在开始介绍这场波澜壮阔的卷王赛道之前，我们先来聊聊AI这个领域如何做研究。一般人先想到的就是**模型架构**，这个是最核心的创新，比如NN和Transformer架构就属于这一类。还有一类就是**训练方法**，这一类的代表就是两阶段学习（PTM+SFT），强化学习和对比学习等。最后一类是关于**训练数据**的，这部分工作不难但是很费时间，对业界的贡献也是很显著的，典型的代表就Image领域的ImageNet和VL领域CoCo。本文主要围绕着新的**模型架构**和**训练方法**来介绍。下图是VL模型迭代年鉴，从2020年ViT开始，到2022年BeiTv3这个集大成者的模型出现。

![MLLM_VL.svg](../images/MLLM_VL.svg)

本文将按照顺序讲解CLIP，ViLT，ALBEF，VLMo到BeiTv3这个五个模型。接下来，我们介绍另一个简单且高效的模型CLIP。

### CLIP：让模型同时理解图像和文字

在开始讨论CLIP前，我们先仔细想想围绕着处理两种模态——图像（Image）和文本（Text）有哪些任务？分别是：

* 检索任务(Retrieval, image <- -> text)
* 图像描述生成(Captioning, image -> text)
* 图像生成任务(Generation, text -> image)
* 视觉问答(Visual question answering, image+text -> text)
* 多模态分类(Multimodal classification, image+text -> label)
* 增强理解/生成任务(Better understanding/generation, image+text -> label/text)

如果仔细看一下这些任务，无一例外，都需要AI模型同时理解图片和文字（VL Understanding）。回想一下Transformer处理NLP的步骤，把Token映射到一个语义空间里，取概率最大的Token输出。前文提到2020年ViT已经成功的解决了第一个问题——舍弃CNN，用Transformer这个特征提取器统一处理Image和Text问题。那么现在第二个问题来了，需要用一个方法把Image和Text，通过跨模态学习，将图像和文本在统一的语义空间中表示，能够让AI模型同时理解Image和Text,即完成下图中的多模态融合层（Modality Interacton）的工作。

![MLLM_two_tower.svg](../images/MLLM_two_tower.svg)

随便提一下，上图这个模型也称为**双塔模型**，擅长处理VL Understanding问题。2021年，CLIP的出现，解决了这个问题。

#### CLIP架构

CLIP，全名Contrastive Language-Image Pre-training，OpenAI团队提出的一种图像+语言的多模态模型。其核心就是通过对比学习（Contrastive Learning），让图像和文字可以通过同一种方式表达它们的含义，这样AI模型就能直接判断一张图和一句话是否在说同一件事。你可以把整个训练过程理解为一个“图文匹配游戏”：每次给模型一组配对的图片和文字。
举个例子，一只躺在草地上的狗的图片和一句话“a photo of a dog”，这是一对正例，而这张图与不相关的句子，比如“An owl on the snow”，则构成负例。每一对正负例中的图和文都会被分别编码成相同维度的“向量”，然后模型会比较它们之间的距离：

* 对于正例，目标是让它们距离更近
* 对于负例，目标是让距离更远

经过大量这样的训练，使CLIP学会判断一张图和一句话是否表达相同含义的能力。对比学习的整个过程如下图所示：

![MLLM_CL.png](../images/MLLM_CL.png)

在CLIP的训练过程中，模型会对成千上万对图像和文字进行匹配训练，让相关的图像和文字对的数值向量靠近，不相关的则相互远离。这样一来，CLIP可以理解文字和图片的含义并进行匹配，从而实现「看图识物」或「图文对比」的效果。下图展示了CLIP模型的核心工作原理和应用流程。

![MLLM_CLIP.png](../images/MLLM_CLIP.png)

整个过程分为三个主要步骤：

1. 对比学习预训练
   图中所示，文本编码器(Text Encoder)将每个文本转换为嵌入$T_1,T_2,...,T_N$​，图像编码器(Image Encoder)则将每个图像转换为相应的嵌入$I_1,I_2,...,I_N$。然后，模型通过对比学习来优化图像和文本的嵌入，使得正确的图文对的相似性最高，即对角线$I_1T_1, I_2T_2,...,I_NT_N$。
2. 基于文本标签创建分类器
   这一步把分类标签改为描述性的句子，例如把dog转换为a photo of dog。通过消融实验，这个简单的转化能提高不少准确率。
3. 零样本预测
   这一步就是做预测了，即CLIP用于做Zero-shot的分类任务了。输入一张图片dog，图像编码器将其向量化，然后把这个dog图片的向量和所有文本向量进行对比，找到与这个图片向量最接近的文本向量，从而实现Zero-shot分类。

CLIP原理非常简单，很容易想到。在OpenAI团队做这个之前，就有人做一样的事情，但是效果远远没有CLIP好。其核心原因就是OpenAI喜欢大力出奇迹，又发挥了一把Scalling Law的威力。最后提一句，CLIP把Text和Image对比学习的办法，简称为ITC（Image-Text Constrastive），这个简称在后续章节中还会用到。

接下来，我们来介绍ViLT，这个模型也是ViT同一个团队出品的。

### ViLT：用Transformer统一Visual Embedding层

ViLT(Vision-and-Language Transformer)是一种多模态模型，旨在通过Transformer架构统一视觉和语言的嵌入层。由ViT团队提出，主要特点是去掉了传统视觉任务中的卷积操作和区域特征提取器，直接将图像和文本输入到Transformer中进行联合建模。
回到文章最初提到的**双塔模型**部分的VE（Visual Embedding）部分，2021年前，VL预训练比较依赖图像特征提起，比如，区域监督（如目标检测）与卷积架构（如ResNet）。这样造成Visual Embedding效率很低，模型在这部分花了大量的时间。ViLT的做法简单粗暴，直接用patch projection代替区域监督和卷积的工作。结果模型速度飞快，准确率比起传统方法下降一点，还能接收。结果详见下图：

![MLLM_ViLT_1.png](../images/MLLM_ViLT_1.png)

来看一下ViLT的架构图，一个典型的**双塔模型**，左下是典型的Lang/Text Embedding, 右下是Visual Embedding。MI层，用的是传统的Transformer Encoder。这边聊一下最上层的ITM（Image Text Matching）和MLM（Masked Language Modeling）方法。ITM方法就是对图像文本对（Image-Text Pair）做预测，如果图片和文字描述一致，就返回True。MLM就是BERT模型里采用方法一样，有兴趣的读者，请读一下《大语言模型：通往通用人工智能之路》，这里就不做过多的陈述了。

![MLLM_ViLT_2.png](../images/MLLM_ViLT_2.png)

不管是ViLT还是CLIP，从技术上来说，都不复杂，只是把别的领域用的好的方法应用到VL这个领域。接下来要介绍的模型就开始啃硬骨头了，开始真正意义上去更改模型架构以达成较好的效果。

### ALBeF



## 参考文献

1. [AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929)
2. [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020)
3. [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/pdf/2102.03334)
4. [一文搞懂多模态：BeiT-3之前的14个多模态+4个周边原理解读](https://zhuanlan.zhihu.com/p/633946545)

