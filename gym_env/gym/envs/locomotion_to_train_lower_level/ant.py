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

"""Wrapper for creating the ant environment."""

import math
import numpy as np
import mujoco_py
import os

from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.locomotion import mujoco_goal_env

from gym.envs.locomotion import goal_reaching_env
from gym.envs.locomotion import maze_env
from gym.envs.locomotion import offline_env
from gym.envs.locomotion import wrappers

from copy import deepcopy

GYM_ASSETS_DIR = os.path.join(
    os.path.dirname(mujoco_goal_env.__file__),
    'assets')

class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
  """Basic ant locomotion environment."""
  FILE = os.path.join(GYM_ASSETS_DIR, 'ant.xml')

  def __init__(self, file_path=None, expose_all_qpos=False,
               expose_body_coms=None, expose_body_comvels=None, non_zero_reset=False):
    if file_path is None:
      file_path = self.FILE

    self._expose_all_qpos = expose_all_qpos
    self._expose_body_coms = expose_body_coms
    self._expose_body_comvels = expose_body_comvels
    self._body_com_indices = {}
    self._body_comvel_indices = {}
    self.current_reward = 0

    self._non_zero_reset = non_zero_reset

    mujoco_env.MujocoEnv.__init__(self, file_path, 5)
    utils.EzPickle.__init__(self)

  @property
  def physics(self):
    # Check mujoco version is greater than version 1.50 to call correct physics
    # model containing PyMjData object for getting and setting position/velocity.
    # Check https://github.com/openai/mujoco-py/issues/80 for updates to api.
    if mujoco_py.get_version() >= '1.50':
      return self.sim
    else:
      return self.model

  def _step(self, a):
    return self.step(a)

  def step(self, a):
    # a += np.random.uniform(-1,1,8)
    # print(a)
    xposbefore = self.get_body_com("torso")[0]
    pos_before = deepcopy(self.get_body_com("torso")[:2])
    self.do_simulation(a, self.frame_skip)
    xposafter = self.get_body_com("torso")[0]
    pos_after = deepcopy(self.get_body_com("torso")[:2])
    # curr_reward = (xposafter - xposbefore) / self.dt
    # curr_reward = round(1. * (np.linalg.norm(pos_after)-np.linalg.norm(pos_before)), 2)
    if np.linalg.norm(pos_after) - np.linalg.norm(pos_before) > 0.04:
      curr_reward = 0
    else:
      curr_reward = -1

    self.pos_after = deepcopy(pos_after)
    self.pos_before = deepcopy(pos_before)

    self.current_reward = curr_reward
    ctrl_cost = .5 * np.square(a).sum()
    contact_cost = 0.5 * 1e-3 * np.sum(
        np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
    survive_reward = 1.0
    # reward = curr_reward - ctrl_cost - contact_cost + survive_reward
    state = self.state_vector()
    notdone = np.isfinite(state).all() \
        and state[2] >= 0.2 and state[2] <= 1.0
    done = not notdone
    ob = self._get_obs()
    return ob, curr_reward, done, dict(
        reward_forward=curr_reward,
        reward_ctrl=-ctrl_cost,
        reward_contact=-contact_cost,
        reward_survive=survive_reward,
        position=deepcopy(pos_after),
        pos_before=deepcopy(pos_before),
        pos_after=deepcopy(pos_after))

  def _get_obs(self):
    self._expose_all_qpos = 1
    # No cfrc observation.
    if self._expose_all_qpos:
      obs = np.concatenate([
          self.physics.data.qpos.flat[:15],  # Ensures only ant obs.
          self.physics.data.qvel.flat[:14],
      ])
    else:
      obs = np.concatenate([
          self.physics.data.qpos.flat[2:15],
          self.physics.data.qvel.flat[:14],
      ])

    if self._expose_body_coms is not None:
      for name in self._expose_body_coms:
        com = self.get_body_com(name)
        if name not in self._body_com_indices:
          indices = range(len(obs), len(obs) + len(com))
          self._body_com_indices[name] = indices
        obs = np.concatenate([obs, com])

    if self._expose_body_comvels is not None:
      for name in self._expose_body_comvels:
        comvel = self.get_body_comvel(name)
        if name not in self._body_comvel_indices:
          indices = range(len(obs), len(obs) + len(comvel))
          self._body_comvel_indices[name] = indices
        obs = np.concatenate([obs, comvel])
    return obs.copy()

  def get_cond_reward(self, action, target, info=None):
    # Normalize lcode from -1 to 1
    target = 2.0 * target - 1.0

    pos_before = info['pos_before'].copy()
    pos_after = info['pos_after'].copy()

    x1, y1 = pos_before
    x2, y2 = pos_after

    x = np.degrees(np.arctan2( y2-y1, x2-x1 ))
    # x = (y2-y1) / (x2-x1+0.000001)
    if x > 0:
      slope_chosen = x
    else:
      slope_chosen = (360 + x)
    # Normalize from -1 to 1
    slope_chosen = (slope_chosen/180.) - 1

    reward = ((-1000.*0.1*(np.square(np.linalg.norm(slope_chosen-target))))/20 + 5.)# * 0.2
    return reward

  def get_direction_taken(self, action, info=None):
    # Normalize lcode from -1 to 1

    pos_before = info['pos_before'].copy()
    pos_after = info['pos_after'].copy()

    x1, y1 = pos_before
    x2, y2 = pos_after

    # degree = np.degrees(np.arctan2( y2, x2 ))
    degree = np.degrees(np.arctan2( y2 - y1, x2 - x1 ))
    # x = (y2-y1) / (x2-x1+0.000001)
    if degree > 0:
      slope_chosen = degree
    else:
      slope_chosen = (360 + degree)
    # Normalize from -1 to 1
    slope_chosen = (slope_chosen/180.) - 1

    # reward = ((-1000.*0.1*(np.square(np.linalg.norm(slope_chosen-target))))/20 + 5.)# * 0.2
    return slope_chosen

  def _get_achieved_goal(self):
    return np.array([self.pos_after[0], self.pos_after[1], 0]).copy()

  def _reset_model(self):
    qpos = self.init_qpos #+ self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
    qvel = self.init_qvel #+ self.np_random.randn(self.model.nv) * .1

    if self._non_zero_reset:
      """Now the reset is supposed to be to a non-zero location"""
      reset_location = self._get_reset_location()
      qpos[:2] = reset_location

    # Set everything other than ant to original position and 0 velocity.
    qpos[15:] = self.init_qpos[15:]
    qvel[14:] = 0.
    qpos[:2] = [0,0]#self._rowcol_to_xy([1,3])
    self.set_state(qpos, qvel)
    return self._get_obs()

  def viewer_setup(self):
    self.viewer.cam.distance = self.model.stat.extent * 0.8
    body_id = self.sim.model.body_name2id('torso')
    lookat = self.sim.data.body_xpos[body_id]
    for idx, value in enumerate(lookat):
        self.viewer.cam.lookat[idx] = value
    # self.viewer.cam.fixedcamid = 3
    # self.viewer.cam.type = const.CAMERA_FIXED
    # self.viewer.cam.azimuth = 140.
    self.viewer.cam.elevation = -50.

  def get_xy(self):
    pos = self.physics.data.qpos[:2]
    return pos.copy()

  def get_reward(self):
    return self.current_reward

  def set_xy(self, xy):
    qpos = np.copy(self.physics.data.qpos)
    qpos[0] = xy[0]
    qpos[1] = xy[1]
    qvel = self.physics.data.qvel
    self.set_state(qpos, qvel)
  

class GoalReachingAntEnv(goal_reaching_env.GoalReachingEnv, AntEnv):
  """Ant locomotion rewarded for goal-reaching."""
  BASE_ENV = AntEnv

  def __init__(self, goal_sampler=goal_reaching_env.disk_goal_sampler,
               file_path=None,
               expose_all_qpos=False, non_zero_reset=False, eval=False, reward_type='sparse', **kwargs):
    goal_reaching_env.GoalReachingEnv.__init__(self, goal_sampler, eval=eval, reward_type=reward_type)
    AntEnv.__init__(self,
                    file_path=file_path,
                    expose_all_qpos=expose_all_qpos,
                    expose_body_coms=None,
                    expose_body_comvels=None,
                    non_zero_reset=non_zero_reset)

class AntMazeEnv(maze_env.MazeEnv, GoalReachingAntEnv, offline_env.OfflineEnv):
  """Ant navigating a maze."""
  LOCOMOTION_ENV = GoalReachingAntEnv

  def __init__(self, goal_sampler=None, expose_all_qpos=True,
               reward_type='sparse',
               *args, **kwargs):
    if goal_sampler is None:
      goal_sampler = lambda np_rand: maze_env.MazeEnv.goal_sampler(self, np_rand)
    maze_env.MazeEnv.__init__(
        self, *args, manual_collision=False,
        goal_sampler=goal_sampler,
        expose_all_qpos=expose_all_qpos,
        reward_type=reward_type,
        **kwargs)
    offline_env.OfflineEnv.__init__(self, **kwargs)

    ## We set the target foal here for evaluation
    # self.set_target()
  
  def set_target(self, target_location=None):
    return self.set_target_goal(target_location)

  def seed(self, seed=0):
      mujoco_env.MujocoEnv.seed(self, seed)

def make_ant_maze_env(**kwargs):
  env = AntMazeEnv(**kwargs)
  return wrappers.NormalizedBoxEnv(env)
  
