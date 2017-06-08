# NLP入门

入门自然语言处理并非需要一头杀入苦海，私以为这是个非常前沿且理论很艰深的方向，但快速入门的最佳方式依然是从感性的角度认识这个领域，也就是拿个例子按照教程自己跑跑。如果想在工程学上做到技术全面，可以试试自己爬取，清洗，建模，将模型封装成服务，甚至，在前端展示。

anyway，这个库属于俺这种愚蠢的初学者。

### 关于word2vec

- 为啥需要将词向量化？

	简单来说，最初的one-hot representation存在两个问题：1.维度太高且过于稀疏 2.没法度量词与词之间的相似度
	
- 模型——分布的假设

	- count-based method：以LSA为代表
	- predictive method：以neural probabilistic language models为代表

- word2vec的两个模型

	1. Continuous Bag-of-Words model
	2. Skip-Gram model