# RLlib-with-Dict-State
A minimal example demonstrating how to use RLlib with states which are presented as dictionaries. 
Let's get into it.

# Why this tutorial?

Machine learning can be divided into (sometimes overlapping) branches of :
* Unsupervised learning,
* Supervised learning, and
* Reinforcement learning (RL).

Unlike the un/supervised learning, which are extensively used in industry, RL is still
not that often in use. Although this has been the case so far, with recent developments
in making RL more feasible, this seems to change. For a review of refer to 
[sds](https://ai.googleblog.com/2021/07/reducing-computational-cost-of-deep.html).

One of the prominent libraries for scalable reinforcement learning, both in academia and industry, is
industry. In this library, you can easily train your agent with different RL algorithms, which are 
available in torch or tensorflow, for discrete/continues actions, and occasional multi-agent support. 
RLlib offers an easy way to implement new RL algorithms, your custom neural network, or has a 
