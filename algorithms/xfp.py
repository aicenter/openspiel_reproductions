from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

from open_spiel.python.algorithms import fictitious_play
from open_spiel.python.algorithms import exploitability
import pyspiel

import os
from pathlib import Path
import csv
import time
import wandb

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 100000, "Number of training iterations.")
flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_integer("players", 2, "Number of players")
flags.DEFINE_integer("logfreq", 100, "How often to print the exploitability")

wandb.init(project="rci-tests")
wandb.run.summary["Solver"] = "XFP"
wandb.config.update(flags.FLAGS)

def main(argv):
    game = pyspiel.load_game(FLAGS.game, {"players": pyspiel.GameParameter(FLAGS.players)})
    solver = fictitious_play.XFPSolver(game)

    for i in range(FLAGS.iterations):
        solver.iteration()

        if i % FLAGS.logfreq == 0:
            conv = exploitability.exploitability(game, solver.average_policy())
            wandb.log({"Iteration": i, 'NashConv': conv})

if __name__ == "__main__":
    app.run(main)