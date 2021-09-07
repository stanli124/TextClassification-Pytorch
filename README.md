# TextClassification-Pytorch

该代码仓库使用pytorch实现了几种文本分类模型，参考了[keras实现](https://github.com/ShawnyXiao/TextClassification-Keras)

实现的模型包括：**TextCNN**, **TextRNN**, **TextBiRNN**, **TextAttBiRNN**, **HAN**, **RCNN**, **RCNNVariant**。

# 环境说明

torch 1.4.0

python 3.6.9

gensim 3.8.0

torchtext 0.6.0

numpy 1.18.5

pandas 1.1.1

# 包说明

所有模型都在model目录下HAN、RCNN、RCNNVariant三个模型中都有news_classification目录，该目录下包含pytorch实现；这三个模型的数据集采用了头条的新闻数据集，没有采用原来的数据集，这三个模型能直接运行。toutiao_cat_data是原始未处理数据集，在model目录下。

除了上述三个模型，其它模型使用的数据集是使用 [keras实现](https://github.com/ShawnyXiao/TextClassification-Keras) 中的数据集合，且其pytorch实现和keras实现在一个目录下。



README-ZH是keras实现的md文档

修改测试
