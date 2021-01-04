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

import os
from pathlib import Path
import csv
import time

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 100000, "Number of training iterations.")
flags.DEFINE_integer("num_traversals", 40, "Number of traversals/games")
flags.DEFINE_string("game", "leduc_poker", "Name of the game")
flags.DEFINE_integer("players", 2, "Number of players")
flags.DEFINE_integer("logfreq", 100, "How often to print the exploitability")
flags.DEFINE_string("logname", "deep_cfr", "Results output filename prefix")
flags.DEFINE_string("logdir", "logs", "Directory for log files")

def log(start, end, iter_logged, i, conv):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    elapsed = "{:0>2}:{:0>2}:{:.1f}".format(int(hours), int(minutes), seconds)
    logging.info("Iteration: {iteration} | " \
            "{n_iter} iterations took {elapsed} | " \
            "exploitability: {conv:.5}".format(iteration=i, n_iter= i + 1 - iter_logged, elapsed=elapsed, conv=conv))

def loginit(log_prefix):
    i = 0
    while os.path.exists("{log_prefix}_{i}.csv".format(log_prefix=log_prefix, i=i)):
        i += 1
    log_filename = "{log_prefix}_{i}.csv".format(log_prefix=log_prefix, i=i)

    with open(log_filename, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "exploitability"])

    return log_filename

def main(argv):
    Path(FLAGS.logdir).mkdir(parents=True, exist_ok=True)
    log_prefix = os.path.join(FLAGS.logdir, FLAGS.logname)
    log_filename = loginit(log_prefix)
    
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
            batch_size_advantage=128,
            batch_size_strategy=1024,
            memory_capacity=1e7,
            policy_network_train_steps=400,
            advantage_network_train_steps=20,
            reinitialize_advantage_networks=False)
        sess.run(tf.global_variables_initializer())
    
        start = time.time()
        iter_logged = 0

        outer_iter = int(FLAGS.iterations / FLAGS.logfreq)

        for i in range(outer_iter):
            _, advantage_losses, policy_loss = deep_cfr_solver.solve()
            
            average_policy = policy.tabular_policy_from_callable(game, deep_cfr_solver.action_probabilities)
            conv = exploitability.nash_conv(game, average_policy)

            with open(log_filename, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([i * FLAGS.logfreq, conv])
            end = time.time()
            log(start, end, iter_logged * FLAGS.logfreq, i * FLAGS.logfreq, conv)
            iter_logged = i
            start = end

if __name__ == "__main__":
    app.run(main)