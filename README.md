# NLP入门

入门自然语言处理并非需要一头杀入苦海，私以为这是个非常前沿且理论很艰深的方向，但快速入门的最佳方式依然是从感性的角度认识这个领域，也就是拿个例子按照教程自己跑跑。如果想在工程学上做到技术全面，可以试试自己爬取，清洗，建模，将模型封装成服务，甚至，在前端展示。

anyway，这个库属于俺这种愚蠢的初学者。

### Abstract

情感分析（观点挖掘）是指用自然语言过程和文本挖掘等技术将一些主观的信息进行分类，目前广泛应用于客户评论、调查反馈、社交媒体、推荐系统等领域。本项目旨在介绍自然语言处理的基本概念，在此基础之上使用电商网站客户评论数据，讲解基于词向量的学习进行情感分析的案例。

可能用到的模型：

- word2vec（必选）
- 常用的有监督学习模型（必选）
- CNN（待定）
- RNN（待定）

推荐要读的论文：

1. Mikolov, Tomas, et al. "Efficient estimation of word representations in vector space." arXiv preprint arXiv:1301.3781 (2013).
2. Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). "Learning Word Vectors for Sentiment Analysis." The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
3. Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).
4. Socher, Richard, et al. "Recursive deep models for semantic compositionality over a sentiment treebank." Proceedings of the conference on empirical methods in natural language processing (EMNLP). Vol. 1631. 2013.

技术上主要使用python实现，如果遇到非要用stanford NLP搞定的，那就一定要写java，虽然也有python的API，总的来说不全。


### 关于玄乎的word2vec

- 为啥需要将词向量化？

	简单来说，最初的one-hot representation存在两个问题：1.维度太高且过于稀疏 2.没法度量词与词之间的相似度
	
- 模型——分布的假设

	- count-based method：以LSA为代表
	- predictive method：以neural probabilistic language models为代表

- word2vec的两个模型

	1. Continuous Bag-of-Words model
	2. Skip-Gram model