from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 100000, "Number of training iterations.")
flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_integer("players", 2, "Number of players")
flags.DEFINE_integer("logfreq", 100, "logging frequency")
flags.DEFINE_string("project", "openspiel", "project name")
flags.DEFINE_boolean("no_wandb", False, "Disables Weights & Biases")

def main(argv):
    if not FLAGS.no_wandb:
        import wandb
        wandb.init(project=FLAGS.project)
        wandb.config.update(flags.FLAGS)

    game = pyspiel.load_game(FLAGS.game, {"players": pyspiel.GameParameter(FLAGS.players)})
    cfr_solver = cfr.CFRSolver(game)

    for i in range(FLAGS.iterations):
        cfr_solver.evaluate_and_update_policy()
        
        if i % FLAGS.logfreq == 0:
            exp = exploitability.exploitability(game, cfr_solver.average_policy())
            if not FLAGS.no_wandb:
                wandb.log({"Iteration": i, 'Exploitability': exp})

            logging.info("Iteration: {} Exploitability: {}".format(i, exp))

if __name__ == "__main__":
    app.run(main)
        