# BPR-Torch
BPR算法是基础的推荐算法，在我入门推荐系统时很容易理解BPR算法，但是当我想实现完整的推荐算法时，却困难重重，如何加载数据，如何进行负采样，如何写评价指标函数，让我很烦恼。
参照王翔老师的NGCF算法 [Tensorflow版](https://github.com/xiangwang1223/neural_graph_collaborative_filtering) 和 [PyTorch版](https://github.com/huangtinglin/NGCF-PyTorch) 
自己实现一个完整的BPR推荐算法框架。

我对此进行了一些改进，想把整个代码封装成一个BPR类，但是由于cuda和并行评价的冲突，导致BPR类代码在类的思想有点瑕疵，还需再改进

- 运行方法
python main.py
