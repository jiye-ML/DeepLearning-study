## 基础知识

![](study_softmax/不同数字可能对应特征权重.png)
![](study_softmax/softmax_计算公式.png)




## 杂谈

### 为什么神经网络的最后一层都是softmax

* 因为softmax使用了sigmoid函数，再加上归一化，让预测出来的结果是一个概率，方便后面的计算。