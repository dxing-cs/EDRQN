import random
import numpy as np
from envs.pursuit.world import World


class Prey:
    def __init__(self, world):
        self.world = world
        self.x = None
        self.y = None
        self.agent_idx = None
        self.frozen = False

    @staticmethod
    def is_predator():
        return False

    @staticmethod
    def is_prey():
        return True

    def is_frozen(self):
        return self.frozen

    def reset(self, agent_idx):
        self.agent_idx = agent_idx
        self.x, self.y = self.world.place_prey_randomly(self)
        self.frozen = False

    def get_loc(self):
        return self.x, self.y

    def get_agent_idx(self):
        return self.agent_idx

    def plan_to_move_to(self, action):
        if action == Pursuit.UP:
            return self.x, (self.y - 1) % self.world.grid_height
        elif action == Pursuit.DOWN:
            return self.x, (self.y + 1) % self.world.grid_height
        elif action == Pursuit.LEFT:
            return (self.x - 1) % self.world.grid_width, self.y
        elif action == Pursuit.RIGHT:
            return (self.x + 1) % self.world.grid_width, self.y
        elif action == Pursuit.NOOP:
            return self.x, self.y
        else:
            raise ValueError

    def move_to(self, new_x, new_y):
        self.world.remove(self.x, self.y)
        self.x, self.y = new_x, new_y
        self.world.place(self.x, self.y, self)

    @staticmethod
    def get_action(noop_prob=0.2):
        r = random.random()
        if 0 <= r < (1 - noop_prob) / 4:
            return Pursuit.UP
        elif (1 - noop_prob) / 4 <= r < (1 - noop_prob) / 2:
            return Pursuit.RIGHT
        elif (1 - noop_prob) / 2 <= r < (1 - noop_prob) * 3 / 4:
            return Pursuit.DOWN
        elif (1 - noop_prob) * 3 / 4 <= r < 1 - noop_prob:
            return Pursuit.LEFT
        else:
            return Pursuit.NOOP


class Predator(Prey):
    def __init__(self, world):
        super(Predator, self).__init__(world)
        self.caught_prey_idx = -1  # if this predator caught any prey, this value will be set to the index of that prey

    @staticmethod
    def is_predator():
        return True

    @staticmethod
    def is_prey():
        return False

    def reset(self, agent_idx=None, total_prey=None):
        self.agent_idx = agent_idx
        self.x, self.y = self.world.place_predator_randomly(self)
        self.frozen = False
        self.caught_prey_idx = -1


class Pursuit:
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    NOOP = 4

    def __init__(self, **kwargs):
        # init world
        self.world = World()

        # init predator and prey
        self.num_predator = 2
        self.num_prey = 4
        self.predator_list = [Predator(self.world) for _ in range(self.num_predator)]
        self.prey_list = [Prey(self.world) for _ in range(self.num_prey)]

        self.step_idx_in_episode = 0
        self.max_step_in_episode = kwargs['max_step'] - 1 if 'max_step' in kwargs.keys() else 299
        self.reset()

    def reset(self):
        self.world.init_world()
        for predator_idx, predator in enumerate(self.predator_list):
            predator.reset(predator_idx)
        for prey_idx, prey in enumerate(self.prey_list):
            prey.reset(prey_idx)
        self.step_idx_in_episode = 0
        return self._get_n_obs()

    def step(self, n_action):
        self._move_predator(n_action)
        self._move_prey()

        n_obs = self._get_n_obs()
        n_done = self._get_n_done()
        n_reward = self._get_n_reward()
        info = {
            'prey_idx': self.predator_list[0].caught_prey_idx
            if len(set(predator.caught_prey_idx for predator in self.predator_list)) == 1 else -1
        }
        self.step_idx_in_episode += 1
        return n_obs, n_reward, n_done, info

    def _move_predator(self, n_action):
        normal_order = random.randint(0, 1)
        randomized_predator_list = self.predator_list if normal_order else reversed(self.predator_list)
        randomized_action_list = n_action if normal_order else reversed(n_action)

        for predator, action in zip(randomized_predator_list, randomized_action_list):
            plan_to_move_to = predator.plan_to_move_to(action)
            if not self.world.is_occupied(*plan_to_move_to) and not predator.is_frozen():
                predator.move_to(*plan_to_move_to)
                x, y = predator.x, predator.y
                if self.world.has_prey_at((x - 1) % self.world.grid_width, y):
                    prey = self.world.fetch((x - 1) % self.world.grid_width, y)
                    self._caught_prey(predator, prey)
                if self.world.has_prey_at((x + 1) % self.world.grid_width, y):
                    prey = self.world.fetch((x + 1) % self.world.grid_width, y)
                    self._caught_prey(predator, prey)
                if self.world.has_prey_at(x, (y - 1) % self.world.grid_height):
                    prey = self.world.fetch(x, (y - 1) % self.world.grid_height)
                    self._caught_prey(predator, prey)
                if self.world.has_prey_at(x, (y + 1) % self.world.grid_height):
                    prey = self.world.fetch(x, (y + 1) % self.world.grid_height)
                    self._caught_prey(predator, prey)

    @staticmethod
    def _caught_prey(predator, prey):
        prey.frozen = True
        predator.frozen = True
        predator.caught_prey_idx = prey.agent_idx

    def _move_prey(self):
        for prey in self.prey_list:
            action = prey.get_action()
            plan_to_move_to = prey.plan_to_move_to(action)
            if not self.world.is_occupied(*plan_to_move_to) and \
                    not self.world.is_out_of_fence(*plan_to_move_to, prey.agent_idx) and \
                    not prey.frozen:
                prey.move_to(*plan_to_move_to)

    def _get_n_obs(self):
        n_obs = []
        for predator in self.predator_list:
            # self's coordinate
            obs = [predator.x, predator.y]
            # other predator's coordinate
            for other_predator in self.predator_list:
                if other_predator != predator:
                    obs.append(other_predator.x)
                    obs.append(other_predator.y)
            # closest prey's coordinate
            closest_prey = self._get_closest_prey(predator)
            obs.append(closest_prey.x)
            obs.append(closest_prey.y)

            n_obs.append(np.array(obs, dtype=np.float32))
        return n_obs

    def _get_closest_prey(self, predator):
        closest_prey = None
        closest_dist = -1
        for prey in self.prey_list:
            curr_dist = abs(prey.x - predator.x) + abs(prey.y - predator.y)
            if closest_dist < 0 or curr_dist < closest_dist:
                closest_dist = curr_dist
                closest_prey = prey
        return closest_prey

    def _get_n_done(self):
        return [predator.is_frozen() or self.step_idx_in_episode >= self.max_step_in_episode
                for predator in self.predator_list]

    def _get_n_reward(self):
        num_frozen_predator = sum(predator.is_frozen() for predator in self.predator_list)

        if self.step_idx_in_episode >= self.max_step_in_episode:  # out of time
            return [-1. for _ in self.predator_list]
        elif num_frozen_predator == 0:                            # wandering
            return [0. for _ in self.predator_list]
        elif num_frozen_predator == 1:                            # one predator captures a prey
            return [
                1. if predator.is_frozen() else 0.
                for predator in self.predator_list
            ]
        else:                                                     # two predators capture preys, might be a same one
            num_frozen_prey = sum(prey.is_frozen() for prey in self.prey_list)
            return [
                1. if num_frozen_prey == 1 else -1.
                for _ in self.predator_list
            ]

    def render(self):
        self.world.render()

    def get_obs_size(self):
        return len(self._get_n_obs()[0])

    @staticmethod
    def get_act_size():
        return 5
