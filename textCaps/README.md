[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

# textCaps
本目录包含了胶囊网络的相关工作

### **textCaps/胶囊网络原理**
一份对胶囊网络原理的详细概述，仅限内部交流，请勿公开。

### **textCaps/reference**
包含了关于胶囊网络的相关论文和资料，可以辅助对于《胶囊网络原理》的阅读。


### **textCaps/textCaps**
一个简单的胶囊网络文本分类模型，模型结构如下：

- 卷积层： 与textcnn的首层类似，本模型设置n-gram的窗口大小为3
- 胶囊层1：设置了20个维度为16的胶囊，本模型使用的是共享权重的胶囊
- 胶囊层2：设置了10个维度为16的胶囊
- 全连接层：将胶囊压平成160维的特征向量，维度为2的全连接层，输出正负类的预判概率

\-----------------------------------------------------------------

同样利用涉黄的训练集训练模型，用自构建的testdataset数据作为性能评估结果如下：

- **acc:  0.800000**

- **time: 1.600521**

其中分类错误的文本：

- testdataset/0/1.txt
- testdataset/0/10.txt
- testdataset/0/9.txt
- testdataset/1/1.txt

与其它深度学习模型类似，由于训练集小说居多，训练的模型对小说十分敏感，容易将小说误判。本模型的胶囊层用keras实现，现有的几种tensorflow版本的实现复用性不高，胶囊层实现参考了苏剑林的版本： [https://github.com/bojone/Capsule](https://github.com/bojone/Capsule)
