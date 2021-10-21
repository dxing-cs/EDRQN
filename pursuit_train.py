import itertools
import datetime
import os
import torch
import numpy as np
import logging
import json
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from envs.pursuit.Pursuit_env import Pursuit
from agent.EDRQN import EDRQN
from utilities.util_function import set_seed

########################################################
# initialize the environment and config file
########################################################
with open('config_pursuit_edrqn.json', 'r') as fin:
    config = json.load(fin)
torch.set_num_threads(1)
set_seed(config['seed'])
env = Pursuit(max_step=config['max_step'])
config['obs_size'] = env.get_obs_size()
config['act_size'] = env.get_act_size()

if config['save_result']:
    config['log_path'] = f"log/pursuit/{config['cur_agent']}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if not os.path.exists(config['log_path']):
        os.makedirs(config['log_path'])
        os.makedirs(config['log_path'] + '/incremental')
    config_content = str(config)
    config['summary_writer'] = SummaryWriter(config['log_path'])
    config['summary_writer'].add_text('log_dir', config['log_path'])
    config['summary_writer'].add_text('config', config_content)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(config['log_path'], 'train.log'))
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

config['caught_prey'] = [0 for _ in range(4)]
config['steps_in_recent_100_episodes'] = deque([0 for _ in range(100)])
config['success_rate'] = deque([0 for _ in range(100)])

########################################################
# initialize agents and play the game
########################################################
agent_0 = EDRQN(config, agent_idx=0)
agent_1 = EDRQN(config, agent_idx=1)

for config['current_episode'] in itertools.count():
    obs = env.reset()
    done = [False]
    config['current_episode_step'] = 0
    if config['render']:
        env.render()

    while not all(done):
        act_0 = agent_0.pick_action(obs[0], on_training=True)
        act_1 = agent_1.pick_action(obs[1], on_training=True)

        act = [act_0, act_1]
        next_obs, rew, done, info = env.step(act)

        agent_0.store_experience(obs[0], act_0, rew[0], next_obs[0], done[0])
        agent_1.store_experience(obs[1], act_1, rew[1], next_obs[1], done[1])

        agent_0.update_prev_hidden(done[0])
        agent_1.update_prev_hidden(done[1])

        if config['current_episode'] >= 500:
            agent_0.learn()
            agent_1.learn()

        obs = next_obs
        config['current_episode_step'] += 1
        config['current_global_step'] += 1
        if config['render']:
            env.render()

    # log results
    if info['prey_idx'] >= 0:
        config['caught_prey'][info['prey_idx']] += 1
    config['steps_in_recent_100_episodes'].popleft()
    config['steps_in_recent_100_episodes'].append(config['current_episode_step'])
    running_avg_steps = sum(config['steps_in_recent_100_episodes']) / 100.0
    config['success_rate'].popleft()
    config['success_rate'].append(1 if info['prey_idx'] >= 0 else 0)
    running_win_rate = sum(config['success_rate']) / 100.0

    print(f"episode: {config['current_episode']} \t\t "
          f"caught_prey: {info['prey_idx'] + 1 if info['prey_idx'] >= 0 else 0} \t\t "
          f"caught_prey_dist: {config['caught_prey']} \t\t "
          f"req_step: {config['current_episode_step']} \t\t "
          f"avg_step: {running_avg_steps} \t\t "
          f"win_rate: {running_win_rate}")

    if config['save_result']:
        config['summary_writer'].add_scalar(
            'pursuit/step',
            config['current_episode_step'],
            config['current_episode']
        )
        config['summary_writer'].add_scalar(
            'pursuit/entropy',
            -np.sum(np.array(config['caught_prey']) / (sum(config['caught_prey']) + 1e-8) * np.log(np.array(config['caught_prey']) / (sum(config['caught_prey']) + 1e-8) + 1e-8)),
            config['current_episode']
        )
        config['summary_writer'].add_scalar(
            'pursuit/win_rate',
            1 if info['prey_idx'] >= 0 else 0,
            config['current_episode']
        )

        logger.info(
            '{}\t{}\t{}\t{}\t{}\t{}'.format(
                config['current_episode'],
                info['prey_idx'] + 1 if info['prey_idx'] >= 0 else 0,
                config['caught_prey'],
                config['current_episode_step'],
                running_avg_steps,
                running_win_rate
            )
        )

    if config['save_result'] and config['current_episode'] % 1000 == 0:
        config['summary_writer'], sw = None, config['summary_writer']
        torch.save([agent_0.state_dict(), agent_1.state_dict(), config],
                   f"{config['log_path']}/incremental/ep_{config['current_episode']}.pt")
        config['summary_writer'] = sw
