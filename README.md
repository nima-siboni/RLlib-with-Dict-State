# RLlib-with-Dict-State
A minimal example demonstrating how to use RLlib with states which are presented as dictionaries. 
Let's get into it.

# Why this tutorial?

Unlike the un/supervised learning, which are extensively used in industry, RL is still
not that often in utilized, in spite [its potential](https://www.sciencedirect.com/science/article/pii/S0004370221000862). 
Nevertheless, there are recent developments in making RL easier to train and more reliable to use
(For a review of refer to [sds](https://ai.googleblog.com/2021/07/reducing-computational-cost-of-deep.html). 
In light of these developments, you might want to try RL for solving your challenges. In this tutorial, a minimal but 
easily extendable use case is explained which could serve as a blueprint for your project. 

# Main ingredients of your RL-project
As deep-RL practitioner, you need:

* an agent, which has one or many neural network to make decisions,
* an environment with which the agent can interact, and
* an algorithm which can train the neural network(s) of the agent based on its interaction with the environments.

In the following, we explore these ingredients, having in mind that this is a tutorial for RL-practitioners.
## RL-algorithm
The latter item of the above list, i.e. the RL-algorithm, is conceptually the most complicated part of the reinforcement learning. 
If you are not a researcher of RL methodology, you are not commonly expected to devise one such algorithm. Instead, you 
should know which one would be a better method for your problem at hand. After choosing the appropriate algorithm you can
find the efficient/stable implementation of you algorithm in one the existing RL libraries.

There are [many of these libraries](https://neptune.ai/blog/the-best-tools-for-reinforcement-learning-in-python),
and we choose [RLlib](https://docs.ray.io/en/master/rllib.html) which is one of the prominent libraries for scalable RL,
both in academia and industry. So far so good, a high quality algorithm is ready to use for training your agent. Easy!

## Environtment
The environment is very specific to your problem and this is the part which should be implemented by you or your team.
Your domain expertise goes into designing states, actions and most importantly the rewards. In this tutorial, we rewrite
 an OpenAI Gym environment, i.e. ```CartPole``` , with a modification: for the state, our custom environment returns a dictionary 
 instead of a numpy array. This gives us the opportunity to:

* create and register a custom environment,
* create custom models for handling dictionaries within RLlib.

### Why states with dictionary?

Both of these two are highly specific to (for?) your problem, and therefore it is very likely that you need to implement that 
by yourself or your team. 

In this tutorial, we focus on how to handle custom environment which have states in format of python dictionary
Th
One of the prominent libraries for scalable reinforcement learning, both in academia and industry, is
industry. In this library, you can easily train your agent with different RL algorithms, which are 
available in torch or tensorflow, for discrete/continues actions, and occasional multi-agent support. 
RLlib offers an easy way to implement new RL algorithms, your custom neural network, or has a 
