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

"""Policy gradient agents trained and evaluated on Kuhn Poker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import policy_gradient

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(1e6), "Number of train episodes.")
flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_enum("loss_str", "rpg", ["a2c", "rpg", "qpg", "rm"],
                  "PG loss to use.")
flags.DEFINE_integer("num_hidden", 64, "Hidden units.")
flags.DEFINE_integer("num_layers", 1, "Hidden layers.")
flags.DEFINE_integer("logfreq", int(1e3), "logging frequency")
flags.DEFINE_string("project", "openspiel", "project name")
flags.DEFINE_boolean("no_wandb", False, "Disables Weights & Biases")
flags.DEFINE_integer("batch_size", 16, "Training batch size.")
flags.DEFINE_float("entropy_cost", 0.001, "")
flags.DEFINE_float("critic_learning_rate", 0.01, "")
flags.DEFINE_float("pi_learning_rate", 0.01, "")
flags.DEFINE_integer("num_critic_before_pi", 4, "")



class PolicyGradientPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies):
    game = env.game
    player_ids = [0, 1]
    super(PolicyGradientPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict


def main(_):
  if not FLAGS.no_wandb:
    import wandb
    wandb.init(project=FLAGS.project)
    wandb.config.update(flags.FLAGS)
    wandb.config.update({"solver": FLAGS.loss_str})

  game = FLAGS.game
  num_players = 2

  env_configs = {"players": num_players}
  env = rl_environment.Environment(game, **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  with tf.Session() as sess:
    # pylint: disable=g-complex-comprehension
    hidden_layers_sizes = tuple([int(FLAGS.num_hidden) for _ in range(int(FLAGS.num_layers))])
    agents = [
        policy_gradient.PolicyGradient(
            sess,
            idx,
            info_state_size,
            num_actions,
            loss_str=FLAGS.loss_str,
            hidden_layers_sizes=hidden_layers_sizes,
            batch_size=FLAGS.batch_size,
            entropy_cost=FLAGS.entropy_cost,
            critic_learning_rate=FLAGS.critic_learning_rate,
            pi_learning_rate=FLAGS.pi_learning_rate,
            num_critic_before_pi=FLAGS.num_critic_before_pi) for idx in range(num_players)
    ]
    expl_policies_avg = PolicyGradientPolicies(env, agents)

    sess.run(tf.global_variables_initializer())
    for ep in range(FLAGS.num_episodes):

      if (ep + 1) % FLAGS.logfreq == 0:
        losses = [agent.loss for agent in agents]
        expl = exploitability.exploitability(env.game, expl_policies_avg)
        msg = "-" * 80 + "\n"
        msg += "{}: {}\n{}\n".format(ep + 1, expl, losses)
        logging.info("%s", msg)

        if not FLAGS.no_wandb:
          wandb.log({"Iteration": ep, 'Exploitability': expl})

      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step)
        action_list = [agent_output.action]
        time_step = env.step(action_list)

      # Episode is over, step all agents with final info state.
      for agent in agents:
        agent.step(time_step)


if __name__ == "__main__":
  app.run(main)
