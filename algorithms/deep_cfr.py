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

from absl import app
from absl import flags
from absl import logging
import six

import tensorflow.compat.v1 as tf

from open_spiel.python import policy
from open_spiel.python.algorithms import deep_cfr
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
import pyspiel
import collections

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 10, "Number of training iterations.")
flags.DEFINE_integer("num_traversals", 40, "Number of traversals/games")
flags.DEFINE_integer("batch_size_advantage", 128, "Adv fn batch size")
flags.DEFINE_integer("batch_size_strategy", 1024, "Strategy batch size")
flags.DEFINE_integer("num_hidden", 64, "Hidden units in each layer")
flags.DEFINE_integer("num_layers", 3, "Depth of neural networks")
flags.DEFINE_bool("reinitialize_advantage_networks", False,
                  "Re-init value net on each CFR iter")
flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_integer("players", 2, "Number of players")
flags.DEFINE_integer("logfreq", 1, "How often to print the exploitability")
flags.DEFINE_string("project", "openspiel", "project name")
flags.DEFINE_boolean("no_wandb", False, "Disables Weights & Biases")
flags.DEFINE_float("learning_rate", 1e-3, "Optimizer learning rate")
flags.DEFINE_integer("memory_capacity",
                     10000000, "replay buffer capacity")
flags.DEFINE_integer("policy_network_train_steps",
                     400, "training steps per iter")
flags.DEFINE_integer("advantage_network_train_steps",
                     20, "training steps per iter")


def solve(self):
    """Modified deep-cfr solution logic for online policy evaluation"""
    advantage_losses = collections.defaultdict(list)
    for _ in range(self._num_iterations):
        for p in range(self._num_players):
            for _ in range(self._num_traversals):
                self._traverse_game_tree(self._root_node, p)
            if self._reinitialize_advantage_networks:
                # Re-initialize advantage network for player and train from scratch.
                self.reinitialize_advantage_network(p)
            advantage_losses[p].append(self._learn_advantage_network(p))
        self._iteration += 1
    # Train policy network.
    policy_loss = self._learn_strategy_network()
    return self._policy_network, advantage_losses, policy_loss


def main(argv):
    if not FLAGS.no_wandb:
        import wandb
        wandb.init(project=FLAGS.project)
        wandb.config.update(flags.FLAGS)
        wandb.config.update({"solver": "deep cfr"})

    game = pyspiel.load_game(
        FLAGS.game, {"players": pyspiel.GameParameter(FLAGS.players)})

    with tf.Session() as sess:
        deep_cfr_solver = deep_cfr.DeepCFRSolver(
            sess,
            game,
            policy_network_layers=tuple(
                [FLAGS.num_hidden for _ in range(FLAGS.num_layers)]),
            advantage_network_layers=tuple(
                [FLAGS.num_hidden for _ in range(FLAGS.num_layers)]),
            num_iterations=FLAGS.iterations,
            num_traversals=FLAGS.num_traversals,
            learning_rate=FLAGS.learning_rate,
            batch_size_advantage=FLAGS.batch_size_advantage,
            batch_size_strategy=FLAGS.batch_size_strategy,
            memory_capacity=FLAGS.memory_capacity,
            policy_network_train_steps=FLAGS.policy_network_train_steps,
            advantage_network_train_steps=FLAGS.advantage_network_train_steps,
            reinitialize_advantage_networks=FLAGS.reinitialize_advantage_networks)
        sess.run(tf.global_variables_initializer())

        """Modified deep-cfr solution logic for online policy evaluation"""
        advantage_losses = collections.defaultdict(list)
        for i in range(deep_cfr_solver._num_iterations):
            if i % FLAGS.logfreq == 0:
                average_policy = policy.tabular_policy_from_callable(
                    game, deep_cfr_solver.action_probabilities)
                conv = exploitability.nash_conv(game, average_policy)

                logging.info("Iteration: {} NashConv: {}".format(i, conv))
                if not FLAGS.no_wandb:
                    wandb.log(
                        {"Iteration": i, 'NashConv': conv})

            for p in range(deep_cfr_solver._num_players):
                for _ in range(deep_cfr_solver._num_traversals):
                    deep_cfr_solver._traverse_game_tree(
                        deep_cfr_solver._root_node, p)
                if deep_cfr_solver._reinitialize_advantage_networks:
                    # Re-initialize advantage network for player and train from scratch.
                    deep_cfr_solver.reinitialize_advantage_network(p)
                advantage_losses[p].append(
                    deep_cfr_solver._learn_advantage_network(p))

            # Train policy network.
            policy_loss = deep_cfr_solver._learn_strategy_network()
            deep_cfr_solver._iteration += 1


if __name__ == "__main__":
    app.run(main)
