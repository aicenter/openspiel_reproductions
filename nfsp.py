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
from open_spiel.python.algorithms import nfsp

import os
from pathlib import Path
import csv
import time

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_train_episodes", int(3e6), "Number of training episodes.")
flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_integer("players", 2, "Number of players")
flags.DEFINE_string("logname", "nfsp", "Results output filename prefix")
flags.DEFINE_string("logdir", "logs", "Directory for log files")

flags.DEFINE_integer("logfreq", 10000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_list("hidden_layers_sizes", [
    128,
], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")

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

class NFSPPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies, mode):
    game = env.game
    player_ids = [0, 1]
    super(NFSPPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._mode = mode
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

    with self._policies[cur_player].temp_mode_as(self._mode):
      p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict


def main(argv):
    Path(FLAGS.logdir).mkdir(parents=True, exist_ok=True)
    log_prefix = os.path.join(FLAGS.logdir, FLAGS.logname)
    log_filename = loginit(log_prefix)
    
    env_configs = {"players": FLAGS.players}
    env = rl_environment.Environment(FLAGS.game, **env_configs)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
    kwargs = {
        "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
        "epsilon_decay_duration": FLAGS.num_train_episodes,
        "epsilon_start": 0.06,
        "epsilon_end": 0.001,
    }

    with tf.Session() as sess:
        agents = [
            nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
                    FLAGS.reservoir_buffer_capacity, FLAGS.anticipatory_param,
                    **kwargs) for idx in range(FLAGS.players)
        ]
        expl_policies_avg = NFSPPolicies(env, agents, nfsp.MODE.average_policy)

        sess.run(tf.global_variables_initializer())

        start = time.time()
        iter_logged = 0

        for ep in range(FLAGS.num_train_episodes):
            if ep % FLAGS.logfreq == 0:
                losses = [agent.loss for agent in agents]
                expl = exploitability.exploitability(env.game, expl_policies_avg)
                
                end = time.time()
                log(start, end, iter_logged, ep, expl)
                logging.info("Losses: %s", losses)

                with open(log_filename, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([ep, expl])
                
                iter_logged = ep
                start = end

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