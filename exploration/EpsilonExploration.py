import random


class EpsilonExploration:
    def __init__(self, epsilon, action_size):
        self.epsilon = epsilon
        self.action_size = action_size

    def perturb_action(self, action, epsilon=None):
        epsilon = epsilon or self.epsilon
        if random.random() > epsilon:  # exploit
            return action
        else:
            return random.randint(0, self.action_size - 1)

    def perturb_action_with_annealing_noise(self, action,
                                            total_step_to_anneal,
                                            current_step,
                                            total_episode_to_anneal,
                                            current_episode):
        if total_step_to_anneal is not None:
            epsilon = max(1. - (1. - self.epsilon) / total_step_to_anneal * current_step, self.epsilon)
        elif total_episode_to_anneal is not None:
            epsilon = max(1. - (1. - self.epsilon) / total_episode_to_anneal * current_episode, self.epsilon)
        else:
            raise ValueError
        return self.perturb_action(action, epsilon)
