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

import os
from pathlib import Path
import csv

# Temporarily disable TF2 until we update the code.
tf.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_steps", 100000, "Number of iterations")
flags.DEFINE_string("game_name", "kuhn_poker", "Name of the game")
flags.DEFINE_string("solver", "tabular", "Tabular or nn")
flags.DEFINE_float("init_lr", 1.0, "The initial learning rate")
flags.DEFINE_float("lr_decay", .999, "Learnign rate multiplier per timestep")
flags.DEFINE_integer("logfreq", 100, "logging frequency")
flags.DEFINE_string("logname", "ed", "Results output filename prefix")
flags.DEFINE_string("logdir", "../logs", "Directory for log files")

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
  
  game = pyspiel.load_game(FLAGS.game_name)
  solver = exploitability_descent.Solver(game)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    lr = FLAGS.init_lr

    for i in range(FLAGS.num_steps):
      lr *= FLAGS.lr_decay #FLAGS.init_lr / np.sqrt(1 + i)
      conv = solver.step(sess, learning_rate = lr)
      
      if i % FLAGS.logfreq == 0:
        logging.info("Iteration: {} Exploitability: {}".format(i, conv))

        with open(log_filename, 'a') as f:
          writer = csv.writer(f)
          writer.writerow([i, conv])

if __name__ == "__main__":
  app.run(main)
