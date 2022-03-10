# RLlib-with-Dict-State
A minimal example demonstrating the use of RLlib with states which are presented as dictionaries. 

# Why this tutorial?

Despite of
[its potential](https://www.sciencedirect.com/science/article/pii/S0004370221000862) and unlike the un/supervised 
learning, reinforcement learning (RL) is not yet utilized frequently across different industries. The reason behind this infrequency
of usage is commonly attributed to expensive training/tunning processes and unreliable solutions. Nevertheless, recently
there have been efforts to make RL easier to train and  more reliable which could provide a ground for considering RL in 
a wide variety of applications (for a review of these developments please refer to 
a recent paper by Google on 
[reducing the computational cost of RL](https://ai.googleblog.com/2021/07/reducing-computational-cost-of-deep.html)). 
In light of these developments, you might also want to try RL for solving your challenges. In this tutorial, a minimal but 
easily extendable, RL-project is outlined which could serve as a blueprint for your business projects. 

# Main ingredients of your RL-project
As a  deep-RL practitioner, the following three items are the essentials in your project:

* an agent, which has one (or many) neural networks to make decisions,
* an environment with which the agent can interact, and
* an algorithm that can train the neural network(s) of the agent using its interaction with the environments as the 
training data.

In the following, we explore these ingredients, having in mind that this is a tutorial for RL practitioners.
## RL-algorithm
The latter item of the above list, i.e. the RL algorithm, is conceptually the most complicated part of reinforcement learning. If you are not a researcher of the RL methodology, you are not commonly expected to
devise one such algorithm. Instead, you should know the classical and state-of-art algorithms, and also have an
educated guess or experience to know which algorithm could be potentially a better method for your problem. 
How to develop or choose an RL algorithm requires a dedicated review which is not the focus of the current article. 

After choosing the appropriate algorithm, one can find an efficient/stable implementation of it in one of the existing RL 
libraries. There are [many of these libraries](https://neptune.ai/blog/the-best-tools-for-reinforcement-learning-in-python),
and  in this tutorial we choose [RLlib](https://docs.ray.io/en/master/rllib.html). 
RLlib is one of the prominent libraries for scalable RL, both in academia and industry, and has many of the state of art
RL-algorithm implemented. 

So far so good! A high-quality algorithm is ready to be used for training your agents. Easy!

## Environment
The environment is very specific to your problem and this is the part which is commonly implemented by you or your team.
Your domain expertise goes into designing states, actions, and most importantly the rewards. As an environment,
in this tutorial, we rewrite an OpenAI Gym environment, i.e. ```CartPole``` , with a slight modification: for the
state, our custom environment returns a dictionary instead of a NumPy array. Without getting into a full implementation
of an environment, the proposed modification enables us to practice:

* the creation and registration of a custom environment, and
* the creation of custom neural network models for handling dictionaries within RLlib. 

### Why states with ```dictionary``` structure?

The state or the observation of your environment can be as simple as a small
array (as it is in the case of CartPole environment), or it can be of a much more
complicated data structure, i.e. it can be composed of different audio and visual signals combined 
with many scalars of different nature. As an example of such a compound state, one can consider a self-driving agent 
whose state is composed of camera images, geo-location, peripheral sensory data, audio signals from the
surroundings, etc. Of course, this complexity should be also reflected in the design of the agent's neural
network, e.g. a multi-input neural network where the input images are connected to appropriate neural networks like 
convolutional nets and the audio signal to an LSTM or an attention layer. As the more complicated states are common in 
the practical use of RL, we demonstrate how to use states in the format of a dictionary for this tutorial and deal with the 
associated technical complications arising from this choice.

When a dictionary state is passed to RLlib, its default behavior is to flatten and concatenate all the values of the
dictionary into one big one-dimensional array. Obviously, this is not the most efficient way of handling
the state when the state has a complicated structure. Fortunately, RLlib also allows for restoring the original 
structure and accessing the values of the dictionary separately. This involves taking some extra steps and considering 
some technicalities which are demonstrated in this 
tutorial.

# Let's get started!

First, we create a custom environment based on ```CartPole``` environment of OpenAI Gym, such that the state is a 
dictionary. Then an appropriate custom model is created such that it can get the state without flattening its content. 
Finally, we set up training and the test for the current setup. The implemented environment can be found in 
```cart_pole_with_dict.py``` in the ```environments``` folder.

## Custom environment
The state in the original implementation of the environment is _one_ ```Tuple``` with 4 elements:
* the translational position of the cart,
* the translational velocity of the cart,
* the angular position of the pole, and also
* the angular velocity of the pole.

In the original implementation In our modified version of the environment, we separate the information about the cart
and the pole from each other, and present each of them separately as different keys of a dictionary. In this case, we
have a dictionary with _two_ keys, each of them containing an array of size 2.

To implement this modification, we:
* Change the definition of the ```observation_space``` in ```__init___``` such that it is of type ```spaces.Dict``` 
  where the values of the keys are type ```spaces.Box```. The type ```spaces.Dict``` is the Gym equivalent of python 
dictionaries and ```spaces.Box``` can be considered as an equivalent to python/NumPy lists: 
```python
self.observation_space = spaces.Dict(
    {'cart': spaces.Box(low=np.array([-4.800, -Inf]), high=np.array([4.800, Inf]), shape=(2,), dtype=np.float),
     'pole': spaces.Box(low=np.array([-0.418, -Inf]), high=np.array([0.418, Inf]), shape=(2,), dtype=np.float)
    })
```
* Change the returned state from the ```step``` function according to our newly structured state. Here, you just need 
to create a python dictionary with the keys defined above and the corresponding values:
```python
self.state = {'cart': np.array([x, x_dot]), 'pole': np.array([theta, theta_dot])}
```
* Change the returned object in ```reset``` function:
```python
self.state = {'cart': self.np_random.uniform(low=-0.05, high=0.05, size=(2,)),
              'pole': self.np_random.uniform(low=-0.05, high=0.05, size=(2,))}
```
These are the most important modifications. Some minor modifications are also needed for ```render``` function; For 
these changes, we refer the reader to the provided source code. 

## Custom model
Now that we have an environment with a structured state, we need to use a custom model which can take 
advantage of this structure. For our case, and in general to pass a custom model to RLlib, one 
starts by implementing a subclass of RLlib's ```TFModelV2``` which including implementing some of its abstract methods 
and/or overriding the
existing ones. Note that not all the methods of ```TFModelV2``` are needed for all the RL-algorithms; For the case of 
DQN, we need to implement:
* ```__init__```, and 
* ```forward```. 

The implemented model can be found in ```keras_model.py``` in the ```model``` folder.

### ```__init__``` method
In ```__init__``` method, the network is built using the _original_ observation space. Here, the original observation 
space is a term used in RLlib package referring to the state before being flattened. In other words, the original 
observation still has the structure of the state from the environment (in our case it is a dictionary). We first get
the original space and then access its different key-values pair separately. With these information, we create the
```Input``` layers of our neural network:
```python
original_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
self.cart = tf.keras.layers.Input(shape=original_space['cart'].shape, name="cart")
self.pole = tf.keras.layers.Input(shape=original_space['pole'].shape, name="pole")
```

For RLlib to consider our custom model, we should register the model in our training script, which is done in the next 
section.

## Training of the custom agent in the custom model

With the custom model and the custom environment, we are ready to start our training! The training script which is 
explained below can be found in ```experience_and_learn.py``` in the project root folder.

* initialize ```ray``` with

```python
 ray.init()
```

* Register the custom model by ```register_custom_model``` method, and add it as ```custom_model``` to the training 
config: 

```python
# 1 -- Configuring the agent
# 1.1 -- register the keras model
ModelCatalog.register_custom_model("keras_Q_model", KerasQModel)

# 1.2 -- adding the custom model to the configs
config = dqn.DEFAULT_CONFIG.copy()
config["model"] = {"custom_model": "keras_Q_model", "custom_model_config": {}}
```
* Pass the environment config to the training config:
```python
# 1.3 -- Adding the environment config to the main configs
config["env_config"] = {} # A Dict is expected with the env parameter; in our case env doesn't have any configurations.
```
* Create a ```DQNTrainer``` object with the registered model and the custom environment:
```
# 2 -- Creating the agent
agent = dqn.DQNTrainer(config=config, env=CartPoleWithDict)
```
* Start the training:
```python
# 3 -- Trainings
for _ in range(20):
    result = agent.train()
    print(_, result['episode_reward_mean'])

ray.shutdown()
```
