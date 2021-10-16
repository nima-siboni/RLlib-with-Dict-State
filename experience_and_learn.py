from ray.rllib.agents.dqn import dqn
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.logger import pretty_print

from environment.cart_pole_with_dict import CartPoleWithDict
from model.keras_model import KerasQModel
import ray


# 1.1.0 -- register the keras model
ModelCatalog.register_custom_model("keras_Q_model", KerasQModel)

ray.init()
config = dqn.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
config["model"] = {"custom_model": "keras_Q_model", "custom_model_config": {}}
config["env_config"] = {}
agent = dqn.DQNTrainer(config=config, env=CartPoleWithDict)
for _ in range(10):
    print(_)
    result = agent.train()
    print(pretty_print(result))
