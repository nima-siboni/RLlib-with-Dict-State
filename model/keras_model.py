import tensorflow as tf
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.sample_batch import SampleBatch


class KerasQModel(TFModelV2):
    """Custom model for DQN."""
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(KerasQModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)
        self.original_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        self.cart = tf.keras.layers.Input(shape=self.original_space['cart'].shape, name="cart")
        self.pole = tf.keras.layers.Input(shape=self.original_space['pole'].shape, name="pole")

        # Concatenating the inputs
        concatenated = tf.keras.layers.Concatenate()([self.cart, self.pole])

        # Building the dense layers
        x = concatenated
        neuron_lst = [64, 32, 16, 8, num_outputs]
        for layer_id, nr_neurons in enumerate(neuron_lst):
            x = tf.keras.layers.Dense(nr_neurons, name="dense_layer_" + str(layer_id), activation=tf.nn.relu,
                                      kernel_initializer=normc_initializer(1.0))(x)

        layer_out = x
        self.model = tf.keras.Model([self.cart, self.pole], layer_out)
        self.model.summary()

    def forward(self, input_dict, state, seq_lens):
        """Custom core forward method."""
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(input_dict[SampleBatch.OBS], self.obs_space, "tf")

        inputs = {'cart': orig_obs["cart"], 'pole': orig_obs["pole"]}
        model_out = self.model(inputs)

        return model_out, state
