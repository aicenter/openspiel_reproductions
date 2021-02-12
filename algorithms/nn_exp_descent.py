# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python Exploitability Descent example.

This example uses a neural network to approximate the policy. For a simple
tabular example, see the unit tests for the exploitability_descent algorithm:

```
  solver = exploitability_descent.Solver(game)
  with tf.Session() as session:
    for step in range(num_steps):
      nash_conv = solver.Step(session, learning_rate)
```

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python.algorithms import exploitability_descent
from open_spiel.python import simple_nets
import pyspiel

# Temporarily disable TF2 until we update the code.
tf.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_steps", 100000, "Number of iterations")
flags.DEFINE_string("game_name", "kuhn_poker", "Name of the game")
flags.DEFINE_float("init_lr", 0.1, "The initial learning rate")
flags.DEFINE_float("lr_scale", 1., "Learnign rate multiplier per timestep")
flags.DEFINE_float("regularizer_scale", 0.001,
                   "Scale for L2 regularization of NN weights")
flags.DEFINE_integer("num_hidden", 64, "Hidden units.")
flags.DEFINE_integer("num_layers", 1, "Hidden layers.")
flags.DEFINE_integer("logfreq", 100, "logging frequency")
flags.DEFINE_string("project", "openspiel", "project name")
flags.DEFINE_boolean("no_wandb", False, "Disables Weights & Biases")

def main(argv):
  del argv

  if not FLAGS.no_wandb:
    import wandb
    wandb.init(project=FLAGS.project)
    wandb.config.update(flags.FLAGS)
    wandb.config.update({"solver": "nn ed"})

  # Create the game to use, and a loss calculator for it
  logging.info("Loading %s", FLAGS.game_name)

  if FLAGS.game_name == "goofspiel":
    game = pyspiel.load_game_as_turn_based(
    "goofspiel", {
        "imp_info": pyspiel.GameParameter(True),
        "num_cards": pyspiel.GameParameter(4),
        "points_order": pyspiel.GameParameter("descending")
    })
  else:
    game = pyspiel.load_game(FLAGS.game_name)
    
  loss_calculator = exploitability_descent.LossCalculator(game)

  # Build the network
  num_hidden = FLAGS.num_hidden
  num_layers = FLAGS.num_layers
  layer = tf.constant(loss_calculator.tabular_policy.state_in, tf.float64)
  for _ in range(num_layers):
    regularizer = (tf.keras.regularizers.l2(l=FLAGS.regularizer_scale))
    layer = tf.layers.dense(layer, num_hidden, activation=tf.nn.relu, kernel_regularizer=regularizer)
  
  regularizer = (tf.keras.regularizers.l2(l=FLAGS.regularizer_scale))
  layer = tf.layers.dense(layer, game.num_distinct_actions(), kernel_regularizer=regularizer)
  tabular_policy = loss_calculator.masked_softmax(layer)

  # Build the loss - exploitability descent loss plus regularizer loss
  nash_conv, loss = loss_calculator.loss(tabular_policy)
  loss += tf.losses.get_regularization_loss()

  # Use a simple gradient descent optimizer
  learning_rate = tf.placeholder(tf.float64, (), name="learning_rate")
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  #optimizer = tf.train.AdamOptimizer()
  optimizer_step = optimizer.minimize(loss)

  # Training loop
  with tf.train.MonitoredTrainingSession() as sess:
    for step in range(FLAGS.num_steps):
      nash_conv_value, _ = sess.run([nash_conv, optimizer_step],
          feed_dict={
              learning_rate: FLAGS.init_lr / np.sqrt(step * FLAGS.lr_scale + 1),
          })
      # Optionally log our progress
      if step % FLAGS.logfreq == 0:
        if not FLAGS.no_wandb:
          wandb.log({"Iteration": step, 'NashConv': nash_conv_value})

        logging.info("Iteration: {} NashConv: {}".format(step, nash_conv_value))

if __name__ == "__main__":
  app.run(main)
