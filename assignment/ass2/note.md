# 什么是差分隐私？（Differential privacy）

## GPT

差分隐私是一种数据隐私保护框架，它可以在保护个人隐私的同时允许对数据集进行有意义的统计分析。其核心思想是：

1. 通过添加精心设计的随机噪声（如拉普拉斯或高斯噪声）到查询结果中
2. 确保无论数据集中是否包含任何特定个体的数据，查询结果都几乎不会改变
3. 提供数学上可证明的隐私保证，用参数ε（epsilon）表示隐私保护程度

差分隐私的特点是：
- 可以抵抗各种背景知识攻击
- 提供可量化的隐私预算（privacy budget）
- 允许在隐私和数据实用性之间进行权衡
实现差分隐私的常见机制包括拉普拉斯机制、高斯机制和指数机制等。

## paper

### https://programming-dp.com/ch3.html
- 如果两个数据集在单个个体的数据上有所不同，则认为它们是相邻的
- 实现此查询的差分隐私的最简单方法是在其答案中添加随机噪声。关键挑战是添加足够的噪声以满足差分隐私的定义，但又不能太多以至于答案变得过于嘈杂而无法使用。为了使这个过程更容易，差分隐私领域已经开发了一些基本机制，这些机制精确地描述了应该使用什么类型和多少噪声。其中之一被称为拉普拉斯机制
- Unbounded DP && Bounded DP

### https://programming-dp.com/ch4.html
Three properties of DP
- Sequential composition, similar to associative law of addition
    - epsilon higher, privacy less, the pic more pointier, far away from true result.
- Parallel composition  并行组合
- Post processing  后处理