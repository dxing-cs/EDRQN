import torch
from agent.EDRQN import EDRQN


def get_edrqn_edrqn_1(agent_idx):
    agent_name = 'edrqn_edrqn'
    agent_time_slot = 'sample_1'
    original_agent_idx = 1
    return _get_raw_agent(agent_name, agent_time_slot, original_agent_idx, agent_idx)


def get_edrqn_edrqn_2(agent_idx):
    agent_name = 'edrqn_edrqn'
    agent_time_slot = 'sample_1'
    original_agent_idx = 0
    return _get_raw_agent(agent_name, agent_time_slot, original_agent_idx, agent_idx)


def _get_raw_agent(agent_name, agent_time_slot, original_agent_idx, agent_idx):
    agent_config_path = 'log/pursuit/{}/{}/incremental/config.pt'.format(agent_name, agent_time_slot)
    config = torch.load(agent_config_path)
    agent_type = agent_name.split('_')[original_agent_idx]
    if agent_type == 'edrqn':
        agent = EDRQN(config=config, agent_idx=agent_idx)
    else:
        raise NotImplemented

    agent_param = 'log/pursuit/{}/{}/incremental/agent.pt'.format(agent_name, agent_time_slot)
    agent.load_state_dict(torch.load(agent_param)[original_agent_idx])
    return agent
