# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

"""Adapted from efficient-hrl maze_env.py."""

import os
import tempfile
import xml.etree.ElementTree as ET
import math
import numpy as np
import gym
from copy import deepcopy
import random
from mpi4py import MPI

RESET = R = 'r'  # Reset position.
GOAL = G = 'g'

# Maze specifications for dataset generation
U_MAZE = [[1, 1, 1, 1, 1],
          [1, R, 0, 0, 1],
          [1, 1, 1, 0, 1],
          [1, G, 0, 0, 1],
          [1, 1, 1, 1, 1]]

BIG_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, G, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, G, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, G, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]]

HARDEST_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, R, 0, 0, 0, 1, G, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, G, 0, 1, 0, 0, G, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, G, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, 0, 0, 1, G, 0, G, 1, 0, G, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

# Maze specifications for evaluation
U_MAZE_TEST = [[1, 1, 1, 1, 1],
              [1, R, 0, 0, 1],
              [1, 1, 1, 0, 1],
              [1, G, 0, 0, 1],
              [1, 1, 1, 1, 1]]

BIG_MAZE_TEST = [[1, 1, 1, 1, 1, 1, 1, 1],
                [1, R, 0, 1, 1, 0, 0, 1],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 0, G, 1],
                [1, 1, 1, 1, 1, 1, 1, 1]]

HARDEST_MAZE_TEST = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, R, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1, 0, G, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

# MAZE_INITIAL =      [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

MAZE_INITIAL =      [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

# MAZE_1 =            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

# MAZE_1 =            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

MAZE_1 =            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                     [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
                     [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

# MAZE_1 =            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

class MazeEnv(gym.Env):
  LOCOMOTION_ENV = None  # Must be specified by child class.

  def __init__(
      self,
      maze_size_scaling,
      maze_map=None,
      maze_height=0.5,
      manual_collision=False,
      non_zero_reset=False,
      reward_type='sparse',
      *args,
      **kwargs):
    if self.LOCOMOTION_ENV is None:
      raise ValueError('LOCOMOTION_ENV is unspecified.')

    self._maze_array = None
    self.index = 0
    self.fixed_goal = 1
    self.generate_random_maze = 0
    self.generate_one_room_maze = 0
    self.generate_two_room_maze = 0
    self.generate_three_room_maze = 0
    self.generate_four_room_maze = 1
    self.generate_five_room_maze = 0
    self.generate_six_room_maze = 0
    self.generate_eight_room_maze = 0
    self.gates = [-1 for _ in range(8)]
    self.rank_seed =  np.random.get_state()[1][0]

    self.curr_eps_num_wall_collisions = []

    # maze_map = HARDEST_MAZE_TEST#self.get_maze_array()
    # xml_path = self.LOCOMOTION_ENV.FILE
    # tree = ET.parse(xml_path)
    # worldbody = tree.find(".//worldbody")

    # self._maze_map = maze_map

    self._maze_height = maze_height
    self._maze_size_scaling = maze_size_scaling
    self._manual_collision = 1#manual_collision

    self.target_goal = None

    maze_array = np.array(MAZE_1)
    self._maze_map = maze_array.copy()
    self._maze_array = maze_array[1:,1:-1].copy()
    # # file_path = self.generate_pre_maze(initial=True)

    xml_path = self.LOCOMOTION_ENV.FILE
    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")
    torso_x, torso_y = self._find_robot(initial=True)
    x_pos = (len(self._maze_map)//2) * self._maze_size_scaling
    y_pos = (len(self._maze_map[0])//2) * self._maze_size_scaling
    self.x_pos = x_pos
    self.y_pos = y_pos
    self._init_torso_x = torso_x
    self._init_torso_y = torso_y


    for i in range(len(self._maze_map)):
      for j in range(len(self._maze_map[0])):
        struct = self._maze_map[i][j]
        if struct == 1:  # Unmovable block.
          # Offset all coordinates so that robot starts at the origin.
          ET.SubElement(
              worldbody, "geom",
              name="object_%d_%d" % (i, j),
              pos="%f %f %f" % (j * self._maze_size_scaling - y_pos,
                                (len(self._maze_map)-i-1) * self._maze_size_scaling - x_pos,
                                self._maze_height / 2 * self._maze_size_scaling),
              size="%f %f %f" % (0.5 * self._maze_size_scaling,
                                 0.5 * self._maze_size_scaling,
                                 self._maze_height * self._maze_size_scaling),
              type="box", #mass = "1000000", solimp="1.0 1.0 0.01", solref="0.01 1",
              material="",
              # contype="1",
              # conaffinity="1",
              rgba="0.7 0.5 0.3 1.0",
          )

    torso = tree.find(".//body[@name='torso']")
    geoms = torso.findall(".//geom")

    # _, file_path = tempfile.mkstemp(text=True, suffix='.xml')
    file_path = "/home/vrsystem/gitrep/hacked/gym_env/gym/envs/locomotion/assets/temp_ant_env.xml"
    self.file_path = file_path
    tree.write(file_path)
    # file_path = None

    rank = MPI.COMM_WORLD.Get_rank()
    # xml_path = "/home/vrsystem/gitrep/hacked/gym_env/gym/envs/locomotion/assets/maze_blocks"+str(rank)+".xml"
    # tree1 = ET.parse(xml_path)
    # for i in range(1,110):
    #   body = tree1.find(".//geom[@name='object"+str(i)+"']")
    #   body.attrib['size'] = str(0.5*self._maze_size_scaling)+' '+str(0.5*self._maze_size_scaling)+' '+str(self._maze_height/2*self._maze_size_scaling)
    #   tree1.write(xml_path)

    # _, file_path = tempfile.mkstemp(text=True, suffix='.xml')

    self.LOCOMOTION_ENV.__init__(self, *args, file_path=file_path, non_zero_reset=non_zero_reset, reward_type=reward_type, **kwargs)

  def get_maze_array(self):
    return self._maze_array.copy()

  def if_collision(self):
    for i in range(self.sim.data.ncon):
        contact = self.sim.data.contact[i]
        name1 = self.sim.model.geom_id2name(contact.geom1)
        name2 = self.sim.model.geom_id2name(contact.geom2)
        if name1 is None or name2 is None:
            break
        if (("torso" == name1 and "object" in name2) or ("torso" == name2 and "object" in name2)) or(("torso" == name1 and "object" in name2) or ("torso" == name2 and "object" in name2)):
            # print('contact', i)
            # print('geom1', name1[:6])
            # print('geom2', name2[:6])
            return True
    return False

  def num_walls_collision(self):
    for i in range(self.sim.data.ncon):
        contact = self.sim.data.contact[i]
        name1 = self.sim.model.geom_id2name(contact.geom1)
        name2 = self.sim.model.geom_id2name(contact.geom2)
        if name1 is None or name2 is None:
            break
        if "torso" == name1 and "object" in name2:
            if name2 not in self.curr_eps_num_wall_collisions:
                self.curr_eps_num_wall_collisions.append(name2)
        elif "torso" == name2 and "object" in name1:
            if name1 not in self.curr_eps_num_wall_collisions:
                self.curr_eps_num_wall_collisions.append(name1)
        elif "torso" == name1 and "object" in name2:
            if name2 not in self.curr_eps_num_wall_collisions:
                self.curr_eps_num_wall_collisions.append(name2)
        elif "torso" == name2 and "object" in name1:
            if name1 not in self.curr_eps_num_wall_collisions:
                self.curr_eps_num_wall_collisions.append(name1)

    return len(self.curr_eps_num_wall_collisions)

  def _xy_to_rowcol(self, xy):
    size_scaling = self._maze_size_scaling
    xy = (max(xy[0], 1e-4), max(xy[1], 1e-4))
    return [int(1 + (xy[1]) / size_scaling),
            int(1 + (xy[0]) / size_scaling)]
  
  def _get_reset_location(self,):
    prob = (1.0 - self._np_maze_map) / np.sum(1.0 - self._np_maze_map) 
    prob_row = np.sum(prob, 1)
    row_sample = np.random.choice(np.arange(self._np_maze_map.shape[0]), p=prob_row)
    col_sample = np.random.choice(np.arange(self._np_maze_map.shape[1]), p=prob[row_sample] * 1.0 / prob_row[row_sample])
    reset_location = self._rowcol_to_xy((row_sample, col_sample))
    
    # Add some random noise
    random_x = np.random.uniform(low=0, high=0.5) * 0.5 * self._maze_size_scaling
    random_y = np.random.uniform(low=0, high=0.5) * 0.5 * self._maze_size_scaling

    return (max(reset_location[0] + random_x, 0), max(reset_location[1] + random_y, 0))

  def _rowcol_to_xy(self, rowcol, add_random_noise=False):
    row, col = rowcol
    row = (11+1-(row+1)-1)
    col += 1
    y = row * self._maze_size_scaling - self.x_pos
    x = col * self._maze_size_scaling - self.y_pos
    if add_random_noise:
      x = x + np.random.uniform(low=0, high=self._maze_size_scaling * 0.25)
      y = y + np.random.uniform(low=0, high=self._maze_size_scaling * 0.25)
    return (x, y)

  # def goal_sampler(self, np_random, only_free_cells=True, interpolate=True):
  #   valid_cells = []
  #   goal_cells = []

  #   for i in range(len(self._maze_map)):
  #     for j in range(len(self._maze_map[0])):
  #       if self._maze_map[i][j] in [0, RESET, GOAL] or not only_free_cells:
  #         valid_cells.append((i, j))
  #       if self._maze_map[i][j] == GOAL:
  #         goal_cells.append((i, j))

  #   # If there is a 'goal' designated, use that. Otherwise, any valid cell can
  #   # be a goal.
  #   sample_choices = goal_cells if goal_cells else valid_cells
  #   cell = sample_choices[np_random.choice(len(sample_choices))]
  #   xy = self._rowcol_to_xy(cell, add_random_noise=True)

  #   random_x = np.random.uniform(low=0, high=0.5) * 0.25 * self._maze_size_scaling
  #   random_y = np.random.uniform(low=0, high=0.5) * 0.25 * self._maze_size_scaling

  #   xy = (max(xy[0] + random_x, 0), max(xy[1] + random_y, 0))

  #   return xy

  def generate_pre_maze(self, seed=0, initial=False):
    maze_array = self.generate_maze(seed=seed)
    self._maze_map = maze_array.copy()
    self._maze_array = maze_array.copy()
    self.maze_array = maze_array.copy()
    x_pos = (len(self._maze_map)//2) * self._maze_size_scaling
    y_pos = (len(self._maze_map[0])//2) * self._maze_size_scaling
    num = 0
    # for row in range(len(maze_array)-1):
    #     for col in range(len(maze_array[0])):
    #         num += 1
    #         if maze_array[row][col] == 1:
    #             name = "object"+str(num)+":joint"
    #             object_qpos = self.sim.data.get_joint_qpos(name)
    #             object_qpos[:3] = [(col+1) * self._maze_size_scaling - self.y_pos,
    #                             (len(maze_array)+1-(row+1)-1) * self._maze_size_scaling - self.x_pos,
    #                             self._maze_height / 2 * self._maze_size_scaling]
    #             self.sim.data.set_joint_qpos(name, object_qpos)
    #         else:
    #             name = "object"+str(num)+":joint"
    #             object_qpos = self.sim.data.get_joint_qpos(name)
    #             object_qpos[:3] = [-51.4, -0.55, 1]
    #             self.sim.data.set_joint_qpos(name, object_qpos)
  
  def set_target_goal(self, goal_input=None):
    if goal_input is None:
      self.target_goal = self.goal_sampler(np.random)
    else:
      self.target_goal = goal_input
    
    print ('Target Goal: ', self.target_goal)
    ## Make sure that the goal used in self._goal is also reset:
    self._goal = self.target_goal

  def goal_sampler(self, index = -1):#, x_index_gripper=0, y_index_gripper=0):
    goal = self.get_goal_pos(index)#, x_index_gripper, y_index_gripper)
    # goal = self.get_random_maze_pos()
    self.goal = goal.copy()            
    self.target_goal = goal.copy()
    return goal.copy()

  def get_goal_pos(self, index = -1):#, x_index_gripper=0, y_index_gripper=0):
    if index != -1:
        random.seed(index)
    goal = np.array([0.,0.,0.42])
    # x_index = np.random.randint(2,9)
    # y_index = np.random.randint(2,10)
    # # x_index = random.randrange(2,7)
    # # y_index = random.randrange(2,8)
    # # x_index = random.randrange(2,4)
    # # y_index = random.randrange(2,4)
    # if self._maze_array is not None:
    #     while self._maze_array[x_index,y_index] == 1:
    #         x_index = np.random.randint(2,9)
    #         y_index = np.random.randint(2,10)
    #         # x_index = random.randrange(2,8)
    #         # y_index = random.randrange(2,9)
    #         # x_index = random.randrange(2,4)
    #         # y_index = random.randrange(2,4)
    #     # if self.generate_one_wall_maze or self.generate_two_wall_maze:
    #     #     x_index = random.randrange(6,9)
    #     #     y_index = random.randrange(2,9)
    #     #     while self._maze_array[x_index,y_index] == 1:
    #     #         x_index = random.randrange(6,9)
    #     #         y_index = random.randrange(2,9)

    x_index_gripper = self.x_index_gripper
    y_index_gripper = self.y_index_gripper
    if x_index_gripper < 5 and y_index_gripper < 5:
        x_index = np.random.choice([1,2,3,7,8])
        y_index = np.random.choice([1,2,3,7,8,9])
        while x_index < 5 and y_index < 5:
            x_index = np.random.choice([1,2,3,7,8])
            y_index = np.random.choice([1,2,3,7,8,9])
    elif x_index_gripper > 5 and y_index_gripper < 5:
        x_index = np.random.choice([1,2,3,7,8])
        y_index = np.random.choice([1,2,3,7,8,9])
        while x_index > 5 and y_index < 5:
            x_index = np.random.choice([1,2,3,7,8])
            y_index = np.random.choice([1,2,3,7,8,9])
    elif x_index_gripper < 5 and y_index_gripper > 5:
        x_index = np.random.choice([1,2,3,7,8])
        y_index = np.random.choice([1,2,3,7,8,9])
        while x_index < 5 and y_index > 5:
            x_index = np.random.choice([1,2,3,7,8])
            y_index = np.random.choice([1,2,3,7,8,9])
    else:
        x_index = np.random.choice([1,2,3,7,8])
        y_index = np.random.choice([1,2,3,7,8,9])
        while x_index > 5 and y_index > 5:
            x_index = np.random.choice([1,2,3,7,8])
            y_index = np.random.choice([1,2,3,7,8,9])

    if self.fixed_goal:
        x_index = 9
        y_index = 9
    # x_index = 8
    # y_index = 8
    goal = self._rowcol_to_xy([x_index, y_index])
    # goal = np.array([-14,-14])#self._rowcol_to_xy([x_index, y_index])
    # goal = np.array([12.5,-14])#self._rowcol_to_xy([x_index, y_index])
    # goal = self._rowcol_to_xy([2, 2])
    # goal[0] = 1.08+x_index*0.049
    # goal[1] = 0.5+y_index*0.05
    if index != -1:
        random.seed(self.rank_seed)
    goal = np.array([goal[0], goal[1], 1.0])
    return goal.copy()

  def get_random_maze_pos(self):
    ymin, ymax = self._rowcol_to_xy([0,0])
    xmax, xmin = self._rowcol_to_xy([9,10])
    xx = random.uniform(xmin, xmax)
    yy = random.uniform(ymin, ymax)
    goal = np.array([xx, yy, 0.0])
    return goal.copy()    

  def _find_robot(self, initial=False):
    if initial:
      return np.array([0., 0.])  
    return self.get_xy()
    # structure = self._maze_map
    # size_scaling = self._maze_size_scaling
    # for i in range(len(structure)):
    #   for j in range(len(structure[0])):
    #     if structure[i][j] == RESET:
    #       return j * size_scaling, i * size_scaling
    # raise ValueError('No robot in maze specification.')

  def _is_in_collision(self, pos):
    x, y = pos
    xmin, ymax = self._rowcol_to_xy([0,0])
    xmax, ymin = self._rowcol_to_xy([9,10])
    '''
    xmin, ymax = -13 10.5
    xmax, ymin = 13 -13
    # x:[-13, 13], diff=(13), offset=(0.0)
    # y:[-13, 10.5], diff=(11.75), offset=(-1.25)

    x:[-13, 13], diff=(13), offset=(0.0)
    y:[-13, 10.5], diff=(11.75), offset=(-1.25)
    z:[1.0, 1.0],  diff=(0.), offset=(1.0)

    self.max_u = [13, 11.75, 0.]
    self.action_offset = [0., -1.25, 1.0]
    u = self.action_offset + (self.max_u * u )




    '''
    # self.set_subgoal('subgoal_2', [self._rowcol_to_xy([0,0])[0], self._rowcol_to_xy([0,0])[1], 1.])
    # self.set_subgoal('subgoal_4', [self._rowcol_to_xy([9,10])[0], self._rowcol_to_xy([9,10])[1], 1.])
    if xmin > x or x > xmax or ymin > y or y > ymax:
      return True
    structure = self._maze_array
    size_scaling = self._maze_size_scaling
    for i in range(len(structure)-1):
      for j in range(len(structure[0])):
        if structure[i][j] == 1:
          miny = (len(self._maze_array)+1-(i+1)-1) * size_scaling - size_scaling * 0.5 - self.x_pos
          maxy = (len(self._maze_array)+1-(i+1)-1) * size_scaling + size_scaling * 0.5 - self.x_pos
          minx = (j+1) * size_scaling - size_scaling * 0.5 - self.y_pos
          maxx = (j+1) * size_scaling + size_scaling * 0.5 - self.y_pos
          if minx <= x <= maxx and miny <= y <= maxy:
            return True
    return False

  def check_collision(self, pos):
    x = pos[:, 0]
    y = pos[:, 1]

    xmin, ymax = self._rowcol_to_xy([0,0])
    xmax, ymin = self._rowcol_to_xy([9,10])
    # xmin, xmax = -14, 12.5
    # ymin, ymax = -14, 10
    '''
    xmin, ymax = -13 10.5
    xmax, ymin = 13 -13
    # x:[-13, 13], diff=(13), offset=(0.0)
    # y:[-13, 10.5], diff=(11.75), offset=(-1.25)

    # goal = np.array([-14,-14])#self._rowcol_to_xy([x_index, y_index])
    goal = np.array([12.5,10])#self._rowcol_to_xy([x_index, y_index])

    x:[-13, 13], diff=(13), offset=(0.0)
    y:[-13, 10.5], diff=(11.75), offset=(-1.25)
    z:[1.0, 1.0],  diff=(0.), offset=(1.0)

    self.max_u = [13, 11.75, 0.]
    self.action_offset = [0., -1.25, 1.0]
    u = self.action_offset + (self.max_u * u )




    '''
    # self.set_subgoal('subgoal_2', [self._rowcol_to_xy([0,0])[0], self._rowcol_to_xy([0,0])[1], 1.])
    # self.set_subgoal('subgoal_4', [self._rowcol_to_xy([9,10])[0], self._rowcol_to_xy([9,10])[1], 1.])
    # is_collision_final = np.array([False]*pos.shape[0])
    is_collision_final = np.logical_or(np.logical_or(np.logical_or((xmin > x), (x > xmax)), (ymin > y)), (y > ymax))
    structure = self._maze_array
    size_scaling = self._maze_size_scaling
    for i in range(len(structure)-1):
      for j in range(len(structure[0])):
        if structure[i][j] == 1:
          miny = (len(self._maze_array)+1-(i+1)-1) * size_scaling - size_scaling * 0.5 - self.x_pos
          maxy = (len(self._maze_array)+1-(i+1)-1) * size_scaling + size_scaling * 0.5 - self.x_pos
          minx = (j+1) * size_scaling - size_scaling * 0.5 - self.y_pos
          maxx = (j+1) * size_scaling + size_scaling * 0.5 - self.y_pos
          is_collision = np.logical_and(np.logical_and(np.logical_and((minx <= x), (x <= maxx)), (miny <= y)), (y <= maxy))
          is_collision_final = np.logical_or(is_collision, is_collision_final)
    return np.array(is_collision_final).reshape(-1,1)

  def set_subgoal(self, name, action):
      site_id = self.sim.model.site_name2id(name)
      # sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
      self.sim.model.site_pos[site_id] = action# - sites_offset[0]
      self.sim.model.site_rgba[site_id][3] = 1

  def step(self, action):
    if self._manual_collision:
      old_pos = self.get_xy()
      inner_next_obs, inner_reward, done, info = self.LOCOMOTION_ENV.step(self, action)
      new_pos = self.get_xy()
      # if self._is_in_collision(new_pos):
      new_pos = new_pos.reshape(1,-1)
      if self.check_collision(new_pos):
        self.set_xy(old_pos)
    else:
      inner_next_obs, inner_reward, done, info = self.LOCOMOTION_ENV.step(self, action)
    next_obs = self._get_obs().copy()
    return next_obs, inner_reward, done, info

  def setIndex(self, index):
    self.reset()
    self.index = index
    self.generate_pre_maze(seed=index)
    # self.goal = self.goal_sampler(index).copy()
    x_index = np.random.randint(2,9)
    y_index = np.random.randint(2,10)

    if self.maze_array is not None:
        while self.maze_array[x_index,y_index] == 1:
            # Easy
            x_index = np.random.randint(2,9)
            y_index = np.random.randint(2,10)
    # init_qpos, init_qvel = self.get_init_qpos_qvel()
    qpos = self.init_qpos 
    qvel = self.init_qvel 
    qpos[3:] = self.init_qpos[3:]
    qvel[3:] = 0.
    x_index = 2
    y_index = 2
    # qpos[:2] = np.array([0,0])
    qpos[:2] = self._rowcol_to_xy([x_index, y_index])
    self.set_state(qpos.copy(), qvel.copy())
    self.x_index_gripper = x_index
    self.y_index_gripper = y_index
    self.goal = self.goal_sampler().copy()
    # x:[-13, 13], diff=(13), offset=(0.0)
    # y:[-13, 10.5], diff=(11.75), offset=(-1.25)
    self.set_subgoal('target0', self.goal)
    # self.set_subgoal('subgoal_3', [13, -13, 4.0])
    # Bring ant to initial position
    # self.set_xy(self._rowcol_to_xy([1d,3]))
    # self.set_xy(self._rowcol_to_xy([0,0]))
    # print(self._rowcol_to_xy([1,3]))
    # assert False
    # self.set_xy(self.get_random_maze_pos())
    return self._get_obs().copy()

  def reset_model(self):
    # self.generate_pre_maze(seed=0, initial=True)
    self.x_index_gripper = 0
    self.y_index_gripper = 0
    self.goal = self.goal_sampler().copy()
    self.set_subgoal('target0', self.goal)
    return self.LOCOMOTION_ENV._reset_model(self)


  # def _get_best_next_rowcol(self, current_rowcol, target_rowcol):
  #   """Runs BFS to find shortest path to target and returns best next rowcol. 
  #      Add obstacle avoidance"""
  #   current_rowcol = tuple(current_rowcol)
  #   target_rowcol = tuple(target_rowcol)
  #   if target_rowcol == current_rowcol:
  #       return target_rowcol

  #   visited = {}
  #   to_visit = [target_rowcol]
  #   while to_visit:
  #     next_visit = []
  #     for rowcol in to_visit:
  #       visited[rowcol] = True
  #       row, col = rowcol
  #       left = (row, col - 1)
  #       right = (row, col + 1)
  #       down = (row + 1, col)
  #       up = (row - 1, col)
  #       for next_rowcol in [left, right, down, up]:
  #         if next_rowcol == current_rowcol:  # Found a shortest path.
  #           return rowcol
  #         next_row, next_col = next_rowcol
  #         if next_row < 0 or next_row >= len(self._maze_map):
  #           continue
  #         if next_col < 0 or next_col >= len(self._maze_map[0]):
  #           continue
  #         if self._maze_map[next_row][next_col] not in [0, RESET, GOAL]:
  #           continue
  #         if next_rowcol in visited:
  #           continue
  #         next_visit.append(next_rowcol)
  #     to_visit = next_visit

  #   raise ValueError('No path found to target.')

  # def create_navigation_policy(self,
  #                              goal_reaching_policy_fn,
  #                              obs_to_robot=lambda obs: obs[:2], 
  #                              obs_to_target=lambda obs: obs[-2:],
  #                              relative=False):
  #   """Creates a navigation policy by guiding a sub-policy to waypoints."""

  #   def policy_fn(obs):
  #     # import ipdb; ipdb.set_trace()
  #     robot_x, robot_y = obs_to_robot(obs)
  #     robot_row, robot_col = self._xy_to_rowcol([robot_x, robot_y])
  #     target_x, target_y = self.target_goal
  #     if relative:
  #       target_x += robot_x  # Target is given in relative coordinates.
  #       target_y += robot_y
  #     target_row, target_col = self._xy_to_rowcol([target_x, target_y])
  #     print ('Target: ', target_row, target_col, target_x, target_y)
  #     print ('Robot: ', robot_row, robot_col, robot_x, robot_y)

  #     waypoint_row, waypoint_col = self._get_best_next_rowcol(
  #         [robot_row, robot_col], [target_row, target_col])
      
  #     if waypoint_row == target_row and waypoint_col == target_col:
  #       waypoint_x = target_x
  #       waypoint_y = target_y
  #     else:
  #       waypoint_x, waypoint_y = self._rowcol_to_xy([waypoint_row, waypoint_col], add_random_noise=True)

  #     goal_x = waypoint_x - robot_x
  #     goal_y = waypoint_y - robot_y

  #     print ('Waypoint: ', waypoint_row, waypoint_col, waypoint_x, waypoint_y)

  #     return goal_reaching_policy_fn(obs, (goal_x, goal_y))

  #   return policy_fn

  def generate_maze(self, width=11, height=10, complexity=.2, density=.9, seed=0):
    """Generate a maze using a maze generation algorithm."""
    # Easy: complexity=.1, density=.2
    # Medium: complexity=.2, density=.3
    # Hard: complexity=.4, density=.4
    if self.generate_random_maze:
        ret_array = self.generate_maze_prim(width=width, height=height, complexity=complexity, density=density, seed=seed)
    elif self.generate_three_room_maze:
        ret_array = self.generate_maze_three_room(width=width, height=height, complexity=complexity, density=density, seed=seed)
    elif self.generate_four_room_maze:
        ret_array = self.generate_maze_four_room(width=width, height=height, complexity=complexity, density=density, seed=seed)
    elif self.generate_five_room_maze:
        ret_array = self.generate_maze_five_room(width=width, height=height, complexity=complexity, density=density, seed=seed)
    elif self.generate_six_room_maze:
        ret_array = self.generate_maze_six_room(width=width, height=height, complexity=complexity, density=density, seed=seed)
    elif self.generate_eight_room_maze:
        ret_array = self.generate_maze_eight_room(width=width, height=height, complexity=complexity, density=density, seed=seed)
    # elif self.generate_one_wall_maze:
    #     ret_array = self.generate_maze_one_wall(width=width, height=height, complexity=complexity, density=density, seed=seed)
    # elif self.generate_two_wall_maze:
    #     ret_array = self.generate_maze_two_wall(width=width, height=height, complexity=complexity, density=density, seed=seed)
    else:
        ret_array = None

    # Reset to old seed
    random.seed(self.rank_seed)
    return ret_array

  def generate_maze_prim(self, width=11, height=10, complexity=.3, density=.2, seed=0):
      """Generate a maze using a maze generation algorithm."""
      # Only odd shapes
      # complexity = density
      # Easy:  -.3 0.2
      # Hard: 0.6, 0.9
      random.seed(seed)
      shape = (height, width)#((height // 2) * 2 + 1, (width // 2) * 2 + 1)
      # Adjust complexity and density relative to maze size
      complexity = int(complexity * (5 * (shape[0] + shape[1])))//4 +1  # Size of components
      density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))  # Number of components

      # assert False
      # Build actual maze
      Z = np.zeros((11,11), dtype=int)
      # Fill borders
      # Z[0, :] = Z[-1, :] = 1
      # Z[:, 0] = Z[:, -1] = 1
      # Z[-2,:] = 1
      Z[-1:] = 1
      # Make aisles
      # density = 2
      for i in range(density):
          cap_index = 10
          x, y = (random.randrange(0, shape[0]), random.randrange(0, shape[1]))  # Pick a random position
          while (Z[x, y] or (x+1>=shape[0] or Z[x+1, y]) or (x-1<0 or Z[x-1, y]) or (y+1>=shape[1] or Z[x, y+1]) or 
              (y-1<0 or Z[x, y-1]) or (x-1<0 or y-1<0 or Z[x-1, y-1]) or (x-1<0 or y+1>=shape[1] or Z[x-1, y+1]) or 
              (y-1<0 or x+1>=shape[0] or Z[x+1, y-1]) or (x+1>=shape[0] or y+1>=shape[1] or Z[x+1, y+1])):
              x, y = (random.randrange(0, shape[0]), random.randrange(0, shape[1]))  # Pick a random position
              cap_index -= 1
              if cap_index<0:
                  break
          if cap_index<0:
              continue
          Z[x, y] = 1
          for j in range(complexity):
              neighbours = []
              if y > 1:             neighbours.append(('u',x, y - 2))
              if y < shape[1] - 2:  neighbours.append(('d',x, y + 2))
              if x > 1:             neighbours.append(('l',x - 2, y))
              if x < shape[0] - 2:  neighbours.append(('r',x + 2, y))
              if len(neighbours):
                  direction, x_, y_ = neighbours[random.randrange(0, len(neighbours))]
                  if ((direction == 'u' and Z[x_, y_]==0 and Z[x_, y_+1]==0 and (y_-1<0 or Z[x_, y_-1]==0)) and 
                          ((x_-1<0 or Z[x_-1, y_]==0) and (x_+1>=shape[0] or Z[x_+1, y_]==0)) and
                          ((x_-1<0 or y_-1<0 or Z[x_-1, y_-1]==0) and (x_-1<0 or y_+1>=shape[1] or Z[x_-1, y_+1]==0)) and
                          ((x_+1>=shape[0] or y_-1<0 or Z[x_+1, y_-1]==0) and (x_+1>=shape[0] or y_+1>=shape[1] or Z[x_+1, y_+1]==0)) or
                      (direction == 'd' and Z[x_, y_]==0 and Z[x_, y_-1]==0 and (y_+1>=shape[1] or Z[x_, y_+1]==0)) and
                          ((x_-1<0 or Z[x_-1, y_]==0) and (x_+1>=shape[0] or Z[x_+1, y_]==0)) and
                          ((x_-1<0 or y_-1<0 or Z[x_-1, y_-1]==0) and (x_-1<0 or y_+1>=shape[1] or Z[x_-1, y_+1]==0)) and
                          ((x_+1>=shape[0] or y_-1<0 or Z[x_+1, y_-1]==0) and (x_+1>=shape[0] or y_+1>=shape[1] or Z[x_+1, y_+1]==0)) or
                      (direction == 'l' and Z[x_, y_]==0 and Z[x_+1, y_]==0 and (x_-1<0 or Z[x_-1, y_]==0)) and
                          ((y_-1<0 or Z[x_, y_-1]==0) and (y_+1>=shape[1] or Z[x_, y_+1]==0)) and
                          ((x_-1<0 or y_-1<0 or Z[x_-1, y_-1]==0) and (x_-1<0 or y_+1>=shape[1] or Z[x_-1, y_+1]==0)) and
                          ((x_+1>=shape[0] or y_-1<0 or Z[x_+1, y_-1]==0) and (x_+1>=shape[0] or y_+1>=shape[1] or Z[x_+1, y_+1]==0)) or
                      (direction == 'r' and Z[x_, y_]==0 and Z[x_-1, y_]==0 and (x_+1>=shape[0] or Z[x_+1, y_]==0)) and
                          ((y_-1<0 or Z[x_, y_-1]==0) and (y_+1>=shape[1] or Z[x_, y_+1]==0)) and
                          ((x_-1<0 or y_-1<0 or Z[x_-1, y_-1]==0) and (x_-1<0 or y_+1>=shape[1] or Z[x_-1, y_+1]==0)) and
                          ((x_+1>=shape[0] or y_-1<0 or Z[x_+1, y_-1]==0) and (x_+1>=shape[0] or y_+1>=shape[1] or Z[x_+1, y_+1]==0))):
                      Z[x_, y_] = 1
                      Z[x_ + (x - x_) // 2, y_ + (y - y_) // 2] = 1
                      x, y = x_, y_
      Z[1,3] = 0
      Z[9,9] = 0
      return Z#.astype(int).copy()


  def generate_maze_three_room(self, width=11, height=10, complexity=.2, density=.2, seed=0):
      """Generation a random two room env"""
      # Only odd shapes
      random.seed(seed)
      shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
      # Adjust complexity and density relative to maze size
      complexity = int(complexity * (5 * (shape[0] + shape[1])))  # Number of components
      density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))  # Size of components
      # Build actual maze
      Z = np.zeros(shape, dtype=bool)
      height = 10
      width = 10
      # Fill borders
      # Z[0, :] = Z[-1, :] = 1
      # Z[:, 0] = Z[:, -1] = 1
      # Horizontal Wall
      x = random.randrange(1, height)
      Z[x,:] = 1
      # Vertical Wall
      y1 = -1
      y2 = -1
      x1 = -1
      x2 = -1
      y = random.randrange(2, width-1)
      while y == 3:
          y = random.randrange(2, width-1)
      Z[x+1:,y] = 1
      # Horizontal gates
      if y != 1:
          y1 = random.randrange(1, y)
          Z[x,y1] = 0
      if y != width-1:
          y2 = random.randrange(y+1, width)
          Z[x,y2] = 0
      self.gates[0] = 1.08+x*0.049
      self.gates[1] = 0.5+y1*0.05
      self.gates[2] = 1.08+x*0.049
      self.gates[3] = 0.5+y2*0.05
      # print(self.gates)
      # assert False
      # self.gates[2:3] = [x,y2]
      # Vertical gates
      if x != 1:
          pass
          # x1 = random.randrange(1, x)
          # Z[x1,y] = 0
      else:
          x1 = 1
      if x != height-1:
          x2 = random.randrange(x+1, height)
          Z[x2,y] = 0
      else:
          x2 = height-1
      # self.gates[4] = 1.08+x1*0.049
      # self.gates[5] = 0.5+y*0.05
      self.gates[6] = 1.08+x2*0.049
      self.gates[7] = 0.5+y*0.05
      #Start
      Z[1,3] = 0
      # Goal
      Z[9,9] = 0
      return Z.astype(int).copy()


  def generate_maze_four_room(self, width=11, height=10, complexity=.2, density=.2, seed=0):
      """Generation a ransom four room env"""
      # Only odd shapes
      random.seed(seed)
      shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
      # Adjust complexity and density relative to maze size
      complexity = int(complexity * (5 * (shape[0] + shape[1])))  # Number of components
      density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))  # Size of components
      # Build actual maze
      Z = np.zeros(shape, dtype=bool)
      height = 10
      width = 10
      # Fill borders
      # Z[0, :] = Z[-1, :] = 1
      # Z[:, 0] = Z[:, -1] = 1
      # Horizontal Wall
      x = random.randrange(2, height-1)
      Z[x,:] = 1
      # Vertical Wall
      y1 = -1
      y2 = -1
      x1 = -1
      x2 = -1
      y = random.randrange(2, width-1)
      while y == 3:
          y = random.randrange(2, width-1)
      Z[:,y] = 1
      # Horizontal gates
      if y != 1:
          y1 = random.randrange(1, y)
          Z[x,y1] = 0
      if y != width-1:
          y2 = random.randrange(y+1, width)
          Z[x,y2] = 0
      self.gates[0] = 1.08+x*0.049
      self.gates[1] = 0.5+y1*0.05
      self.gates[2] = 1.08+x*0.049
      self.gates[3] = 0.5+y2*0.05
      # print(self.gates)
      # assert False
      # self.gates[2:3] = [x,y2]
      # Vertical gates
      if x != 1:
          x1 = random.randrange(1, x)
          Z[x1,y] = 0
      else:
          x1 = 1
      if x != height-1:
          x2 = random.randrange(x+1, height)
          Z[x2,y] = 0
      else:
          x2 = height-1

      # Maze 1
      Z = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
                  )

      # Z = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
      #             )

      # Z = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
      #             )

      # Z = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
      #             )

      # Z = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
      #             )

      # Maze 2
      # Z = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      #              [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
      #             )

      # Maze 3
      # Z = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      #              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
      #             )
      # print(x,y1)
      # print(x,y2)
      # print(x1,y)
      # print(x2,y)
      self.gates[4] = 1.08+x1*0.049
      self.gates[5] = 0.5+y*0.05
      self.gates[6] = 1.08+x2*0.049
      self.gates[7] = 0.5+y*0.05
      #Start
      Z[1,3] = 0
      # Goal
      Z[9,9] = 0
      return Z.astype(int).copy()

  def generate_maze_five_room(self, width=11, height=10, complexity=.2, density=.2, seed=0):
      """Generation a ransom four room env"""
      # Only odd shapes
      random.seed(seed)
      shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
      # Adjust complexity and density relative to maze size
      complexity = int(complexity * (5 * (shape[0] + shape[1])))  # Number of components
      density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))  # Size of components
      # Build actual maze
      Z = np.zeros(shape, dtype=bool)
      height = 10
      width = 10
      # Fill borders
      # Z[0, :] = Z[-1, :] = 1
      # Z[:, 0] = Z[:, -1] = 1
      # Horizontal Wall
      x = random.randrange(1, height)
      # Z[x,:] = 1
      # Vertical Wall
      y1 = -1
      y2 = -1
      x1 = -1
      x2 = -1
      y = random.randrange(2, width-1)
      while y == 3:
          y = random.randrange(2, width-1)
      Z[:,y] = 1
      Z[x,:y] = 1
      # Horizontal gates
      if y != 1:
          y1 = random.randrange(1, y)
          Z[x,y1] = 0
      if y != width-1:
          y2 = random.randrange(y+1, width)
          Z[x,y2] = 0
      self.gates[0] = 1.08+x*0.049
      self.gates[1] = 0.5+y1*0.05
      self.gates[2] = 1.08+x*0.049
      self.gates[3] = 0.5+y2*0.05
      # print(self.gates)
      # assert False
      # self.gates[2:3] = [x,y2]
      # Vertical gates
      if x != 1:
          x1 = random.randrange(1, x)
          Z[x1,y:] = 1
      else:
          x1 = 1
      if x != height-1:
          x2 = random.randrange(x+1, height)
          Z[x2,y:] = 1
      else:
          x2 = height-1
      if y != width-1:
          y21 = random.randrange(y+1, width)
          Z[x1,y21] = 0
          y22 = random.randrange(y+1, width)
          Z[x2,y22] = 0

      if x1 < 3:
          Z[x1-1,y] = 0
      else:
          x1_gate = random.randrange(1, x1)
          Z[x1_gate,y] = 0
      if x2 - x1 == 2:
          Z[x2-1,y] = 0
      elif x2 - x1 > 2:
          x2_gate = random.randrange(x1+1, x2-1)
          Z[x2_gate,y] = 0
      if x2 == height-2:
          Z[x2+1,y] = 0
      elif x2 != height-1:
          x3_gate = random.randrange(x2+1, height-1)
          Z[x3_gate,y] = 0
      # print(x,y1)
      # print(x,y2)
      # print(x1,y)
      # print(x2,y)
      self.gates[4] = 1.08+x1*0.049
      self.gates[5] = 0.5+y*0.05
      self.gates[6] = 1.08+x2*0.049
      self.gates[7] = 0.5+y*0.05
      #Start
      Z[1,3] = 0
      # Goal
      Z[9,9] = 0
      return Z.astype(int).copy()

  def generate_maze_six_room(self, width=11, height=10, complexity=.2, density=.2, seed=0):
      """Generation a random six room env"""
      # Only odd shapes
      random.seed(seed)
      shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
      # Adjust complexity and density relative to maze size
      complexity = int(complexity * (5 * (shape[0] + shape[1])))  # Number of components
      density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))  # Size of components
      # Build actual maze
      Z = np.zeros(shape, dtype=bool)
      height = 10
      width = 10
      # Fill borders
      # Z[0, :] = Z[-1, :] = 1
      # Z[:, 0] = Z[:, -1] = 1
      # Horizontal Wall
      x11 = random.randrange(2, height-2)
      if x11+2 == height-1:
          x22 = height-1
      else:
          x22 = random.randrange(x11+2, height-1)
      Z[x11,:] = 1
      Z[x22,:] = 1
      # Vertical Wall
      y1 = -1
      y2 = -1
      x1 = -1
      x2 = -1
      y = random.randrange(2, width-1)
      while y == 3:
          y = random.randrange(2, width-1)
      Z[:,y] = 1
      # Z[x,:y] = 1

      # Vertical gates
      x_min = min(x11, x22)
      x_max = max(x11, x22)
      if x_min != 1:
          x1_gate = random.randrange(1, x_min)
          Z[x1_gate,y] = 0
      else:
          Z[1,y] = 0
      if x_min +1 < x_max:
          x2_gate = random.randrange(x_min+1, x_max)
          Z[x2_gate,y] = 0
      else:
          Z[x_max,y] = 0
      if x_max +1 < height -1:
          x22_gate = random.randrange(x_max+1, height-1)
          Z[x22_gate,y] = 0
      else:
          Z[height-1,y] = 0

      # Horizontal gates
      y1 = random.randrange(1, y)
      Z[x11,y1] = 0
      y2 = random.randrange(y+1, width)
      Z[x11,y2] = 0

      y1 = random.randrange(1, y)
      Z[x22,y1] = 0
      y2 = random.randrange(y+1, width)
      Z[x22,y2] = 0
      # self.gates[0] = 1.08+x*0.049
      # self.gates[1] = 0.5+y1*0.05
      # self.gates[2] = 1.08+x*0.049
      # self.gates[3] = 0.5+y2*0.05
      # print(self.gates)
      # assert False
      # self.gates[2:3] = [x,y2]

      # Vertical gates
      # if x != 1:
      #     x1 = random.randrange(1, x)
      #     Z[x1,y:] = 1
      # else:
      #     x1 = 1
      # if x != height-1:
      #     x2 = random.randrange(x+1, height)
      #     Z[x2,y:] = 1
      # else:
      #     x2 = height-1
      # if y != width-1:
      #     y21 = random.randrange(y+1, width)
      #     Z[x1,y21] = 0
      #     y22 = random.randrange(y+1, width)
      #     Z[x2,y22] = 0

      # if x1 < 3:
      #     Z[x1-1,y] = 0
      # else:
      #     x1_gate = random.randrange(1, x1)
      #     Z[x1_gate,y] = 0
      # if x2 - x1 == 2:
      #     Z[x2-1,y] = 0
      # elif x2 - x1 > 2:
      #     x2_gate = random.randrange(x1+1, x2-1)
      #     Z[x2_gate,y] = 0
      # if x2 == height-2:
      #     Z[x2+1,y] = 0
      # elif x2 != height-1:
      #     x3_gate = random.randrange(x2+1, height-1)
      #     Z[x3_gate,y] = 0


      # if x != 1:
      #     x1 = random.randrange(1, x)
      #     Z[x1,:y] = 1
      # else:
      #     x1 = 1
      # if x != height-1:
      #     x2 = random.randrange(x+1, height)
      #     Z[x2,:y] = 1
      # else:
      #     x2 = height-1
      # if y != width-1:
      #     y21 = random.randrange(1, y)
      #     Z[x1,y21] = 0
      #     y22 = random.randrange(1, y)
      #     Z[x2,y22] = 0

      # if x1 < 3:
      #     Z[x1-1,y] = 0
      # else:
      #     x1_gate = random.randrange(1, x1)
      #     Z[x1_gate,y] = 0
      # if x2 - x1 == 2:
      #     Z[x2-1,y] = 0
      # elif x2 - x1 > 2:
      #     x2_gate = random.randrange(x1+1, x2-1)
      #     Z[x2_gate,y] = 0
      # if x2 == height-2:
      #     Z[x2+1,y] = 0
      # elif x2 != height-1:
      #     x3_gate = random.randrange(x2+1, height-1)
      #     Z[x3_gate,y] = 0
      # # print(x,y1)
      # # print(x,y2)
      # # print(x1,y)
      # # print(x2,y)
      # self.gates[4] = 1.08+x1*0.049
      # self.gates[5] = 0.5+y*0.05
      # self.gates[6] = 1.08+x2*0.049
      # self.gates[7] = 0.5+y*0.05
      #Start
      Z[1,3] = 0
      # Goal
      Z[9,9] = 0
      return Z.astype(int).copy()


  def generate_maze_eight_room(self, width=11, height=10, complexity=.2, density=.2, seed=0):
      """Generation a random six room env"""
      # Only odd shapes
      random.seed(seed)
      shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
      # Adjust complexity and density relative to maze size
      complexity = int(complexity * (5 * (shape[0] + shape[1])))  # Number of components
      density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))  # Size of components
      # Build actual maze
      Z = np.zeros(shape, dtype=bool)
      height = 10
      width = 10
      # Fill borders
      # Z[0, :] = Z[-1, :] = 1
      # Z[:, 0] = Z[:, -1] = 1
      # Horizontal Wall
      x11 = random.randrange(2, height-2)
      if x11+2 == height-1:
          x22 = height-1
      else:
          x22 = random.randrange(x11+2, height-1)
      Z[x11,:] = 1
      Z[x22,:] = 1
      # Vertical Wall
      y1 = -1
      y2 = -1
      x1 = -1
      x2 = -1
      y11 = random.randrange(2, width-2)
      if y11+2 == width-1:
          y22 = width-1
      else:
          y22 = random.randrange(y11+2, width-1)
      Z[:,y11] = 1
      Z[:,y22] = 1
      # Z[x,:y] = 1

      # Vertical gates
      x_min = min(x11, x22)
      x_max = max(x11, x22)
      if x_min != 1:
          x1_gate = random.randrange(1, x_min)
          Z[x1_gate,y11] = 0
      else:
          Z[1,y11] = 0
      if x_min +1 < x_max:
          x2_gate = random.randrange(x_min+1, x_max)
          Z[x2_gate,y11] = 0
      else:
          Z[x_max,y11] = 0
      if x_max +1 < height -1:
          x22_gate = random.randrange(x_max+1, height-1)
          Z[x22_gate,y11] = 0
      else:
          Z[height-1,y11] = 0

      if x_min != 1:
          x1_gate = random.randrange(1, x_min)
          Z[x1_gate,y22] = 0
      else:
          Z[1,y2] = 0
      if x_min +1 < x_max:
          x2_gate = random.randrange(x_min+1, x_max)
          Z[x2_gate,y22] = 0
      else:
          Z[x_max,y11] = 0
      if x_max +1 < height -1:
          x22_gate = random.randrange(x_max+1, height-1)
          Z[x22_gate,y22] = 0
      else:
          Z[height-1,y22] = 0

      # Horizontal gates

      y_min = min(y11, y22)
      y_max = max(y11, y22)
      if y_min != 1:
          y1_gate = random.randrange(1, y_min)
          Z[x11,y1_gate] = 0
      else:
          Z[x11,1] = 0
      if y_min +1 < y_max:
          y2_gate = random.randrange(y_min+1, y_max)
          Z[x11,y2_gate] = 0
      else:
          Z[x11,y_max] = 0
      if y_max +1 < width -1:
          y22_gate = random.randrange(y_max+1, width-1)
          Z[x11,y22_gate] = 0
      else:
          Z[x11,width-1] = 0

      if y_min != 1:
          y1_gate = random.randrange(1, y_min)
          Z[x22,y1_gate] = 0
      else:
          Z[x22,1] = 0
      if y_min +1 < y_max:
          y2_gate = random.randrange(y_min+1, y_max)
          Z[x22,y2_gate] = 0
      else:
          Z[x22,y_max] = 0
      if y_max +1 < width -1:
          y22_gate = random.randrange(y_max+1, width-1)
          Z[x22,y22_gate] = 0
      else:
          Z[x22,width-1] = 0

      #Start
      Z[1,3] = 0
      # Goal
      Z[9,9] = 0
      return Z.astype(int).copy()


  def generate_maze_one_wall(self, width=11, height=10, complexity=.2, density=.2, seed=0):
      """Generation a random one wall env"""
      # Only odd shapes
      random.seed(seed)
      shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
      # Adjust complexity and density relative to maze size
      complexity = int(complexity * (5 * (shape[0] + shape[1])))  # Number of components
      density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))  # Size of components
      # Build actual maze
      Z = np.zeros(shape, dtype=bool)
      height = 10
      width = 10
      # Fill borders
      # Z[0, :] = Z[-1, :] = 1
      # Z[:, 0] = Z[:, -1] = 1
      y1 = -1
      y2 = -1
      x1 = -1
      x2 = -1
      # Horizontal Wall
      # x = random.randrange(1, height)
      x = int(height/2-1)
      Z[x,:] = 1
      # y1 = int(width/2)
      y1 = int(random.randrange(1, width))
      Z[x,y1] = 0
      Z[x,y1-1] = 0
      #Start
      Z[1,3] = 0
      # Goal
      Z[9,9] = 0
      return Z.astype(int).copy()


  def generate_maze_two_wall(self, width=11, height=10, complexity=.2, density=.2, seed=0):
      """Generation a random two wall env"""
      # Only odd shapes
      random.seed(seed)
      shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
      # Adjust complexity and density relative to maze size
      complexity = int(complexity * (5 * (shape[0] + shape[1])))  # Number of components
      density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))  # Size of components
      # Build actual maze
      Z = np.zeros(shape, dtype=bool)
      height = 10
      width = 10
      y1 = -1
      y2 = -1
      x1 = -1
      x2 = -1
      # Fill borders
      # Z[0, :] = Z[-1, :] = 1
      # Z[:, 0] = Z[:, -1] = 1
      # Horizontal Wall
      # x = random.randrange(1, height)
      x1 = int(height/2-2)
      x2 = int(height/2+1)
      Z[x1,:] = 1
      Z[x2,:] = 1
      y1 = int(random.randrange(1, width))
      y2 = int(random.randrange(1, width))
      # y1 = int(width/2-2)
      # y2 = int(width/2+2)
      Z[x1,y1] = 0
      Z[x1,y1-1] = 0
      Z[x2,y2] = 0
      Z[x2,y2-1] = 0
      #Start
      Z[1,3] = 0
      # Goal
      Z[9,9] = 0
      return Z.astype(int).copy()