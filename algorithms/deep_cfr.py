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

flags.DEFINE_integer("iterations", 100000, "Number of training iterations.")
flags.DEFINE_integer("num_traversals", 40, "Number of traversals/games")
flags.DEFINE_integer("batch_size_advantage", 128, "Adv fn batch size")
flags.DEFINE_integer("batch_size_strategy", 1024, "Strategy batch size")
flags.DEFINE_integer("num_hidden", 64, "Hidden units in each layer")
flags.DEFINE_integer("num_layers", 3, "Depth of neural networks")
flags.DEFINE_bool("reinitialize_advantage_networks", False, "Re-init value net on each CFR iter")
flags.DEFINE_string("game", "leduc_poker", "Name of the game")
flags.DEFINE_integer("players", 2, "Number of players")
flags.DEFINE_integer("logfreq", 100, "How often to print the exploitability")
flags.DEFINE_string("project", "openspiel", "project name")
flags.DEFINE_boolean("no_wandb", False, "Disables Weights & Biases")

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
    
    game = pyspiel.load_game(FLAGS.game, {"players": pyspiel.GameParameter(FLAGS.players)})

    with tf.Session() as sess:
        deep_cfr_solver = deep_cfr.DeepCFRSolver(
            sess,
            game,
            policy_network_layers=tuple([FLAGS.num_hidden for _ in range(FLAGS.num_layers)]),
            advantage_network_layers=tuple([FLAGS.num_hidden for _ in range(FLAGS.num_layers)]),
            num_iterations=FLAGS.logfreq,
            num_traversals=FLAGS.num_traversals,
            learning_rate=1e-3,
            batch_size_advantage=FLAGS.batch_size_advantage,
            batch_size_strategy=FLAGS.batch_size_strategy,
            memory_capacity=1e7,
            policy_network_train_steps=400,
            advantage_network_train_steps=20,
            reinitialize_advantage_networks=FLAGS.reinitialize_advantage_networks)
        sess.run(tf.global_variables_initializer())

        outer_iter = int(FLAGS.iterations / FLAGS.logfreq)

        for i in range(outer_iter):
            _, advantage_losses, policy_loss = deep_cfr_solver.solve()
            
            average_policy = policy.tabular_policy_from_callable(game, deep_cfr_solver.action_probabilities)
            conv = exploitability.nash_conv(game, average_policy)
            if not FLAGS.no_wandb:
                wandb.log({"Iteration": i * FLAGS.logfreq, 'NashConv': conv})

            logging.info("Iteration: {} NashConv: {}".format(i * FLAGS.logfreq, conv))

if __name__ == "__main__":
    app.run(main)