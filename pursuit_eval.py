import torch
from envs.pursuit.Pursuit_env import Pursuit
from utilities.util_function import set_seed
from agent.agent_factory_pursuit import *

torch.set_num_threads(1)
set_seed(147264)

agent_0 = get_edrqn_edrqn_1(agent_idx=0)
agent_1 = get_edrqn_edrqn_2(agent_idx=1)

env = Pursuit(max_step=agent_0.config['max_step'])
prey = [0 for _ in range(4)]
req_steps = []
render = False

for ep_idx in range(1000):
    obs = env.reset()
    done = [False]
    current_episode_step = 0
    if render:
        env.render()

    while not all(done):
        act_0 = agent_0.pick_action(obs[0], on_training=True)
        act_1 = agent_1.pick_action(obs[1], on_training=True)
        act = [act_0, act_1]
        next_obs, rew, done, info = env.step(act)

        agent_0.update_prev_hidden(done[0])
        agent_1.update_prev_hidden(done[1])

        obs = next_obs
        current_episode_step += 1
        if render:
            env.render()

    if info['prey_idx'] >= 0:
        prey[info['prey_idx']] += 1
    req_steps.append(current_episode_step)
    print('Episode: {} \t  Captured Preys: {}'.format(ep_idx, prey))

print('=========')
print('caught_prey_dist: ', prey)
print('req_steps: ', sum(req_steps) / 1000.0)
print(sum(prey))
