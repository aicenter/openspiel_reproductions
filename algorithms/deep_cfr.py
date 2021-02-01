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

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 100000, "Number of training iterations.")
flags.DEFINE_integer("num_traversals", 40, "Number of traversals/games")
flags.DEFINE_integer("batch_size_advantage", 128, "Adv fn batch size")
flags.DEFINE_integer("batch_size_strategy", 1024, "Strategy batch size")
flags.DEFINE_bool("reinitialize_advantage_networks", False, "Re-init value net on each CFR iter")
flags.DEFINE_string("game", "leduc_poker", "Name of the game")
flags.DEFINE_integer("players", 2, "Number of players")
flags.DEFINE_integer("logfreq", 100, "How often to print the exploitability")
flags.DEFINE_string("project", "openspiel", "project name")
flags.DEFINE_boolean("no_wandb", False, "Disables Weights & Biases")

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
            policy_network_layers=(16,),
            advantage_network_layers=(16,),
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
                wandb.log({"Iteration": i, 'NashConv': conv})

            logging.info("Iteration: {} NashConv: {}".format(i, conv))

if __name__ == "__main__":
    app.run(main)