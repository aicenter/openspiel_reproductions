# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
# Copyright 2021 Artificial Intelligence Center, Czech Techical University
# Copied and adapted from OpenSpiel (https://github.com/deepmind/open_spiel)
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
import pyspiel

# Temporarily disable TF2 until we update the code.
tf.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_steps", 100000, "Number of iterations")
flags.DEFINE_string("game_name", "kuhn_poker", "Name of the game")
flags.DEFINE_float("init_lr", 1.0, "The initial learning rate")
flags.DEFINE_float("lr_scale", 1., "Learnign rate multiplier per timestep")
flags.DEFINE_integer("logfreq", 100, "logging frequency")
flags.DEFINE_string("project", "openspiel", "project name")
flags.DEFINE_boolean("no_wandb", False, "Disables Weights & Biases")


def main(argv):
    if not FLAGS.no_wandb:
        import wandb
        wandb.init(project=FLAGS.project)
        wandb.config.update(flags.FLAGS)
        wandb.config.update({"solver": "ed"})

    if FLAGS.game_name == "goofspiel":
        game = pyspiel.load_game_as_turn_based(
            "goofspiel", {
                "imp_info": pyspiel.GameParameter(True),
                "num_cards": pyspiel.GameParameter(4),
                "points_order": pyspiel.GameParameter("descending")
            })
    else:
        game = pyspiel.load_game(FLAGS.game_name)
    solver = exploitability_descent.Solver(game)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(FLAGS.num_steps):
            lr = FLAGS.init_lr / np.sqrt(i * FLAGS.lr_scale + 1)
            conv = solver.step(sess, learning_rate=lr)

            if i % FLAGS.logfreq == 0:
                if not FLAGS.no_wandb:
                    wandb.log({"Iteration": i, 'NashConv': conv})

                logging.info("Iteration: {} NashConv: {}".format(i, conv))


if __name__ == "__main__":
    app.run(main)
