import random
from envs.pursuit.draw import draw_grid, fill_cell, draw_circle, write_cell_text
import copy
import numpy as np


class World:
    def __init__(self):
        self.grid_height = 20
        self.grid_width = 20
        self.boundary = 2

        self._world = None
        self.viewer = None
        self._base_img = None
        self.predator_list = []
        self.prey_list = []

    def place(self, x, y, obj):
        self._world[x][y] = obj

    def remove(self, x, y):
        self._world[x][y] = None

    def fetch(self, x, y):
        return self._world[x][y]

    def is_occupied(self, x, y):
        return self._world[x][y] is not None

    def is_out_of_fence(self, x, y, fence_number):
        if fence_number == 0:
            return not self.boundary <= x < self.boundary + 4 or \
                   not self.boundary <= y < self.boundary + 4
        elif fence_number == 1:
            return not self.grid_width - self.boundary - 4 <= x < self.grid_width - self.boundary or \
                   not self.boundary <= y < self.boundary + 4
        elif fence_number == 2:
            return not self.boundary <= x < self.boundary + 4 or \
                   not self.grid_height - self.boundary - 4 <= y < self.grid_height - self.boundary
        elif fence_number == 3:
            return not self.grid_width - self.boundary - 4 <= x < self.grid_width - self.boundary or \
                   not self.grid_height - self.boundary - 4 <= y < self.grid_height - self.boundary

    def has_predator_at(self, x, y):
        return self.is_occupied(x, y) and self._world[x][y].is_predator()

    def has_prey_at(self, x, y):
        return self.is_occupied(x, y) and self._world[x][y].is_prey()

    def init_world(self):
        self._world = [[None for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        self.predator_list = []
        self.prey_list = []

    def _get_an_unoccupied_loc(self, x_min, x_max, y_min, y_max):
        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)
        while self.is_occupied(x, y):
            x = random.randint(x_min, x_max)
            y = random.randint(y_min, y_max)
        return x, y

    def place_predator_randomly(self, obj):
        self.predator_list.append(obj)
        x, y = self._get_an_unoccupied_loc(8, 11, 8, 11)
        self.place(x, y, obj)
        return x, y

    def place_prey_randomly(self, obj):
        self.prey_list.append(obj)
        agent_idx = obj.agent_idx
        x, y = None, None
        if agent_idx == 0:    # upper left fence
            x, y = self._get_an_unoccupied_loc(
                self.boundary,
                self.boundary + 3,
                self.boundary,
                self.boundary + 3
            )
        elif agent_idx == 1:  # upper right fence
            x, y = self._get_an_unoccupied_loc(
                self.grid_width - self.boundary - 4,
                self.grid_width - self.boundary - 1,
                self.boundary,
                self.boundary + 3
            )
        elif agent_idx == 2:  # bottom left fence
            x, y = self._get_an_unoccupied_loc(
                self.boundary,
                self.boundary + 3,
                self.grid_height - self.boundary - 4,
                self.grid_height - self.boundary - 1
            )
        elif agent_idx == 3:  # bottom right fence
            x, y = self._get_an_unoccupied_loc(
                self.grid_width - self.boundary - 4,
                self.grid_width - self.boundary - 1,
                self.grid_height - self.boundary - 4,
                self.grid_height - self.boundary - 1
            )
        self.place(x, y, obj)
        return x, y

    def render(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self._base_img = draw_grid(self.grid_height, self.grid_width, cell_size=50, fill='white', line_color='black')

            for xx in range(self.boundary, self.boundary + 4):
                for yy in range(self.boundary, self.boundary + 4):
                    fill_cell(self._base_img, (xx, yy), cell_size=50, fill=(227, 242, 253), margin=0.05)

            for xx in range(self.grid_width - self.boundary - 4, self.grid_width - self.boundary):
                for yy in range(self.boundary, self.boundary + 4):
                    fill_cell(self._base_img, (xx, yy), cell_size=50, fill=(227, 242, 253), margin=0.05)

            for xx in range(self.boundary, self.boundary + 4):
                for yy in range(self.grid_height - self.boundary - 4, self.grid_height - self.boundary):
                    fill_cell(self._base_img, (xx, yy), cell_size=50, fill=(227, 242, 253), margin=0.05)

            for xx in range(self.grid_width - self.boundary - 4, self.grid_width - self.boundary):
                for yy in range(self.grid_height - self.boundary - 4, self.grid_height - self.boundary):
                    fill_cell(self._base_img, (xx, yy), cell_size=50, fill=(227, 242, 253), margin=0.05)

            self.viewer = rendering.SimpleImageViewer()

        img = copy.copy(self._base_img)
        for idx, predator in enumerate(self.predator_list):
            draw_circle(img, (predator.y, predator.x), cell_size=50, fill=(255, 128, 0), radius=0.1)
            write_cell_text(img, str(idx + 1), (predator.y, predator.x), cell_size=50, fill='white', margin=0.3)
        for idx, prey in enumerate(self.prey_list):
            draw_circle(img, (prey.y, prey.x), cell_size=50, fill=(0, 0, 255), radius=0.1)
            write_cell_text(img, str(idx + 1), (prey.y, prey.x), cell_size=50, fill='white', margin=0.3)
        img = np.asarray(img)
        self.viewer.imshow(img)
        return self.viewer.isopen
