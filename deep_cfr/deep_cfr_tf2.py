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

"""Python Deep CFR example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import six

from open_spiel.python import policy
from open_spiel.python.algorithms import deep_cfr_tf2
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
import pyspiel

import os
from pathlib import Path
import csv

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 100000, "Number of iterations")
flags.DEFINE_integer("num_traversals", 150, "Number of traversals/games")
flags.DEFINE_string("game_name", "leduc_poker", "Name of the game")
flags.DEFINE_integer("logfreq", 100, "logging frequency")

flags.DEFINE_string("logname", "cfr", "Results output filename prefix")
flags.DEFINE_string("logdir", "logs", "Directory for log files")

def loginit(log_prefix):
    i = 0
    while os.path.exists("{log_prefix}_{i}.csv".format(log_prefix=log_prefix, i=i)):
        i += 1
    log_filename = "{log_prefix}_{i}.csv".format(log_prefix=log_prefix, i=i)

    with open(log_filename, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "exploitability"])

    return log_filename


def main(unused_argv):
  Path(FLAGS.logdir).mkdir(parents=True, exist_ok=True)
  log_prefix = os.path.join(FLAGS.logdir, FLAGS.logname)
  log_filename = loginit(log_prefix)

  logging.info("Loading %s", FLAGS.game_name)
  game = pyspiel.load_game(FLAGS.game_name)
  deep_cfr_solver = deep_cfr_tf2.DeepCFRSolver(
      game,
      policy_network_layers=(64, 64, 64, 64),
      advantage_network_layers=(64, 64, 64, 64),
      num_iterations=FLAGS.num_iterations,
      num_traversals=FLAGS.num_traversals,
      learning_rate=1e-3,
      batch_size_advantage=2048,
      batch_size_strategy=2048,
      memory_capacity=1e6,
      policy_network_train_steps=5000,
      advantage_network_train_steps=500,
      reinitialize_advantage_networks=True,
      infer_device="cpu",
      train_device="cpu")

  outer_iter = int(FLAGS.num_iterations / FLAGS.logfreq)
  
  for i in range(outer_iter):
    _, advantage_losses, policy_loss = deep_cfr_solver.solve()
    average_policy = policy.tabular_policy_from_callable(game, deep_cfr_solver.action_probabilities)
    conv = exploitability.nash_conv(game, average_policy)
    logging.info("Iteration: {} Nashconv: {}".format(i, conv))
        
    with open(log_filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([i, conv])

if __name__ == "__main__":
  app.run(main)
