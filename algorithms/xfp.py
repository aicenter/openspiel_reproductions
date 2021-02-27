# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
# Copyright 2021 Artificial Intelligence Center, Czech Techical University
# Copied and adapted from OpenSpiel (https://github.com/deepmind/open_spiel)
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

    if FLAGS.game == "goofspiel":
        game = pyspiel.load_game_as_turn_based(
            "goofspiel", {
                "imp_info": pyspiel.GameParameter(True),
                "num_cards": pyspiel.GameParameter(4),
                "points_order": pyspiel.GameParameter("descending")
            })
    else:
        game = pyspiel.load_game(
            FLAGS.game, {"players": pyspiel.GameParameter(FLAGS.players)})

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
