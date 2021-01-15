from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
import pyspiel

import os
from pathlib import Path
import csv

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 100000, "Number of training iterations.")
flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_integer("players", 2, "Number of players")
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

def main(argv):
    Path(FLAGS.logdir).mkdir(parents=True, exist_ok=True)
    log_prefix = os.path.join(FLAGS.logdir, FLAGS.logname)
    log_filename = loginit(log_prefix)
    
    game = pyspiel.load_game(FLAGS.game, {"players": pyspiel.GameParameter(FLAGS.players)})
    cfr_solver = cfr.CFRSolver(game)

    for i in range(FLAGS.iterations):
        cfr_solver.evaluate_and_update_policy()
        
        if i % FLAGS.logfreq == 0:
            conv = exploitability.exploitability(game, cfr_solver.average_policy())
            logging.info("Iteration: {} Exploitability: {}".format(i, conv))
        
            with open(log_filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([i, conv])

if __name__ == "__main__":
    app.run(main)
        