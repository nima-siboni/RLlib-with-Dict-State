from ray.rllib.agents.dqn import dqn
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.logger import pretty_print

from environment.cart_pole_with_dict import CartPoleWithDict
from model.keras_model import KerasQModel
import ray

ray.init()

# 1 -- Configuring the agent
# 1.1 -- register the keras model
ModelCatalog.register_custom_model("keras_Q_model", KerasQModel)
# 1.2 -- adding the custom model to the configs
config = dqn.DEFAULT_CONFIG.copy()
config["model"] = {"custom_model": "keras_Q_model", "custom_model_config": {}}

# 1.3 -- Adding the environment config to the main configs
config["env_config"] = {}  # A Dict is expected with the env parameter; in our case env
# doesn't have any configurations.

# 2 -- Creating the agent
agent = dqn.DQNTrainer(config=config, env=CartPoleWithDict)

# 3 -- Trainings
for _ in range(20):
    result = agent.train()
    print(_, result['episode_reward_mean'])

ray.shutdown()
