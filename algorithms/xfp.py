from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

from open_spiel.python.algorithms import fictitious_play
from open_spiel.python.algorithms import exploitability
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 100000, "Number of training iterations.")
flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_integer("players", 2, "Number of players")
flags.DEFINE_integer("logfreq", 100, "How often to print the exploitability")
flags.DEFINE_string("project", "openspiel", "project name")
flags.DEFINE_boolean("no_wandb", False, "Disables Weights & Biases")

def main(argv):
    if not FLAGS.no_wandb:
        import wandb
        wandb.init(project=FLAGS.project)
        wandb.config.update(flags.FLAGS)
        wandb.config.update({"solver": "xfp"})
        
    game = pyspiel.load_game(FLAGS.game, {"players": pyspiel.GameParameter(FLAGS.players)})
    solver = fictitious_play.XFPSolver(game)

    for i in range(FLAGS.iterations):
        solver.iteration()

        if i % FLAGS.logfreq == 0:
            exp = exploitability.exploitability(game, solver.average_policy())
            if not FLAGS.no_wandb:
                wandb.log({"Iteration": i, 'Exploitability': exp})
            
            logging.info("Iteration: {}, Exploitability: {}".format(i, exp))

if __name__ == "__main__":
    app.run(main)