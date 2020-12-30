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
import time

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 100000, "Number of training iterations.")
flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_integer("players", 2, "Number of players")
flags.DEFINE_integer("logfreq", 10, "Decays logging frequency over time: (n_iter / (i + 1)) % logfreq == 0")
flags.DEFINE_string("logname", "cfr", "Results output filename prefix")
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
    cfr_solver = cfr.CFRSolver(game)

    start = time.time()
    iter_logged = 0

    for i in range(FLAGS.iterations):
        cfr_solver.evaluate_and_update_policy()
        
        if i % FLAGS.logfreq == 0:
            conv = pyspiel.exploitability(game, cfr_solver.average_policy())
        
            with open(log_filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([i, conv])
            end = time.time()
            log(start, end, iter_logged, i, conv)
            iter_logged = i
            start = end

if __name__ == "__main__":
    app.run(main)
        