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

"""Wrapper for creating the point environment."""

import math
import numpy as np
import mujoco_py
import os

from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.locomotion import mujoco_goal_env

from gym.envs.locomotion import goal_reaching_env
from gym.envs.locomotion import maze_env
from gym.envs.locomotion import wrappers
from copy import deepcopy

MY_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'assets')


class PointEnv(mujoco_env.MujocoEnv, utils.EzPickle):
  FILE = os.path.join(MY_ASSETS_DIR, 'point.xml')

  def __init__(self, file_path=None, expose_all_qpos=False, non_zero_reset=False):
    if file_path is None:
        file_path = self.FILE

    self._expose_all_qpos = expose_all_qpos
    mujoco_env.MujocoEnv.__init__(self, file_path, 1)
    # mujoco_goal_env.MujocoGoalEnv.__init__(self, file_path, 1)
    utils.EzPickle.__init__(self)
    self.drag_len = 0.15*10
    # self.drag_len = 0.6

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

  def step(self, action):
    # action[0] = 1.0 * action[0]
    # action[1] = 1.0 * action[1]
    # action[0] = 2.0 * action[0] - 1.0
    # action[1] = 2.0 * action[1] - 1.0
    # action = 2.0 * action - 1.0
    qpos = np.copy(self.physics.data.qpos)
    qpos_temp = np.copy(self.physics.data.qpos)
    pos_before = deepcopy(qpos[:2])
    
    dx = action[0]
    dy = action[1]
    # qpos[2] += action[1]
    # ori = qpos[2]
    # ori = action[1]
    # Compute increment in each direction.
    # dx = math.cos(ori) * action[0]
    # dy = math.sin(ori) * action[0]
    # Ensure that the robot is within reasonable range.
    # qpos[0] = np.clip(qpos[0] + dx, -22, 22)
    # qpos[1] = np.clip(qpos[1] + dy, -22, 18)

    # Apply actions
    # drag_len = .1
    # drag_theta = action[0]*180+180.0
    # drag_theta = action*180+180.0

    # action_x = drag_len*np.cos(np.radians(drag_theta))*1.0
    # action_y = drag_len*np.sin(np.radians(drag_theta))*1.0
    # qpos[0] = qpos[0] + action_x
    # qpos[1] = qpos[1] + action_y

    pos_temp_before = deepcopy(qpos[:2])
    self.drag_len = 0.15*10
    # self.drag_len = 0.6
    qpos_temp[0] = qpos_temp[0] + dx * self.drag_len
    qpos_temp[1] = qpos_temp[1] + dy * self.drag_len

    pos_temp_after = deepcopy(qpos_temp[:2])

    if np.linalg.norm(pos_temp_after - pos_temp_before) > self.drag_len:
      # Find direction
      x1, y1 = pos_temp_before
      x2, y2 = pos_temp_after

      # degree = np.degrees(np.arctan2( y2, x2 ))
      degree = np.degrees(np.arctan2( y2 - y1, x2 - x1 ))
      # x = (y2-y1) / (x2-x1+0.000001)
      if degree > 0:
        slope_chosen = degree
      else:
        slope_chosen = (360 + degree)
      # Normalize from -1 to 1
      slope_chosen = (slope_chosen/180.) - 1

      # drag_len = .15
      drag_theta = slope_chosen*180+180.0

      action_x = self.drag_len*np.cos(np.radians(drag_theta))*1.0
      action_y = self.drag_len*np.sin(np.radians(drag_theta))*1.0
      qpos[0] = qpos[0] + action_x
      qpos[1] = qpos[1] + action_y
      # pos_after = deepcopy(qpos[:2])
    else:
      qpos = deepcopy(qpos_temp)

    pos_after = deepcopy(qpos[:2])
    self.pos_before = deepcopy(pos_before)
    self.pos_after = deepcopy(pos_after)
    
    # reward = 0
    # if np.linalg.norm(pos_after) > np.linalg.norm(pos_before):
    #   reward = 0.1#np.linalg.norm(pos_after - pos_before)
    # reward = (1. * (np.linalg.norm(pos_after)-np.linalg.norm(pos_before)))
    # print(np.linalg.norm(pos_after)-np.linalg.norm(pos_before))
    if np.linalg.norm(pos_after) - np.linalg.norm(pos_before) > 1.0:
      reward = 0
    else:
      reward = -1
    # reward = 100*(1. * ((pos_after[0])-(pos_before[0])))
    # reward = 0
    # # if pos_after[0] > pos_before[0]:
    # #   reward = round(pos_after[0] - pos_before[0], 2)
    # if pos_after[0] < pos_before[0]:
    #   reward = -1.#round(pos_after[0] - pos_before[0], 2)
    # else:
    #   reward = 100*round(pos_after[0] - pos_before[0], 2)

    qvel = np.copy(self.physics.data.qvel)
    self.set_state(qpos, qvel)
    for _ in range(0, self.frame_skip):
      self.physics.step()
    next_obs = self._get_obs()
    # reward = 0
    done = False
    info = {
            'position': deepcopy(pos_after),
            'pos_before': deepcopy(pos_before),
            'pos_after': deepcopy(pos_after)
        }
    return next_obs, reward, done, info

  def get_next_state_controller(self, original_state, action):
    # print(original_state, action)
    next_state_prev = np.zeros((action.shape[0], 3))
    next_state_new = np.zeros((action.shape[0], 3))
    next_state = np.zeros((action.shape[0], 3))
    dx = action[:,0]
    dy = action[:,1]
    # print(original_state)
    # print(dx, dy)
    next_state_prev[:,0] = original_state[:,0] + dx * self.drag_len
    next_state_prev[:,1] = original_state[:,1] + dy * self.drag_len
    # print(next_state_prev)

    # original_state_temp = original_state.copy()

    greater_cond_flag = np.linalg.norm(next_state_prev[:,:2] - original_state[:,:2], axis=1) > self.drag_len
    greater_cond_flag = greater_cond_flag.reshape(-1,1)

    # Find direction
    x1 = original_state[:,0]
    y1 = original_state[:,1]
    x2 = next_state_prev[:,0]
    y2 = next_state_prev[:,1]

    # degree = np.degrees(np.arctan2( y2, x2 ))
    degree = np.degrees(np.arctan2( y2 - y1, x2 - x1 ))
    # x = (y2-y1) / (x2-x1+0.000001)
    # if degree > 0:
    #   slope_chosen = degree
    # else:
    #   slope_chosen = (360 + degree)
    slope_chosen = np.where(degree>0, degree, 360+degree)

    # Normalize from -1 to 1
    slope_chosen = (slope_chosen/180.) - 1

    # drag_len = .15
    drag_theta = slope_chosen*180+180.0

    action_x = self.drag_len*np.cos(np.radians(drag_theta))*1.0
    action_y = self.drag_len*np.sin(np.radians(drag_theta))*1.0
    
    next_state_new[:,0] = original_state[:,0] + action_x
    next_state_new[:,1] = original_state[:,1] + action_y
    # pos_after = deepcopy(qpos[:2])
    # print('prev', next_state_prev)
    # print('new', next_state_new)
    # print(greater_cond_flag)

    next_state = np.where(greater_cond_flag, next_state_new, next_state_prev)
    # print(next_state)

    # Correcting in case collision occurs
    is_collision = self.check_collision(next_state)
    # assert False
    # print(is_collision)
    # print(original_state)
    # print(next_state)
    # self.set_subgoal('subgoal_2', original_state[0])
    # self.set_subgoal('subgoal_3', next_state[0])
    # for i in range(1000):
    #   self.render()
    # assert False
    next_state = np.where(is_collision, original_state, next_state)
    return next_state.copy(), next_state.copy()


  def get_cond_reward(self, action, target, info=None):
    # Normalize lcode from -1 to 1
    target = 2.0 * target - 1.0
    reward = ((-1000.*0.1*(np.square(np.linalg.norm(action - target))))/20 + 5.)# * 0.2
    return reward

  # def get_direction_taken(self, action, info=None): 
  #     return action.copy()

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

  def _get_obs(self):
    # if self._expose_all_qpos:
    #   return np.concatenate([
    #       self.physics.data.qpos.flat[:3].copy(),  # Only point-relevant coords.
    #       self.physics.data.qvel.flat[:3]].copy())
    return np.concatenate([
        self.physics.data.qpos.flat[:3]])
        # self.physics.data.qvel.flat[:3]])

  def _get_achieved_goal(self):
    return np.array([self.pos_after[0], self.pos_after[1], 0]).copy()

  def _reset_model(self):
    qpos = self.init_qpos #+ self.np_random.uniform(size=self.physics.model.nq, low=-.1, high=.1)
    qvel = self.init_qvel #+ self.np_random.randn(self.physics.model.nv) * .1

    # Set everything other than point to original position and 0 velocity.
    qpos[3:] = self.init_qpos[3:]
    qvel[3:] = 0.
    # qpos[:2] = self._rowcol_to_xy([1,3])
    qpos[:2] = [0,0]
    self.set_state(qpos, qvel)
    return self._get_obs()

  def viewer_setup(self):
    self.viewer.cam.distance = self.model.stat.extent * 1.5
    body_id = self.sim.model.body_name2id('torso')
    lookat = self.sim.data.body_xpos[body_id]
    for idx, value in enumerate(lookat):
        self.viewer.cam.lookat[idx] = value

  def get_xy(self):
    pos = self.physics.data.qpos[:2]
    return pos.copy()

  def set_xy(self, xy):
    qpos = np.copy(self.physics.data.qpos)
    qpos[0] = xy[0]
    qpos[1] = xy[1]
    qvel = self.physics.data.qvel
    self.set_state(qpos, qvel)


class GoalReachingPointEnv(goal_reaching_env.GoalReachingEnv, PointEnv):
  """Point locomotion rewarded for goal-reaching."""
  BASE_ENV = PointEnv

  def __init__(self, goal_sampler=goal_reaching_env.disk_goal_sampler,
               file_path=None,
               non_zero_reset=False,
               reward_type='sparse',
               dataset_url=None,
               expose_all_qpos=False):
    goal_reaching_env.GoalReachingEnv.__init__(self, goal_sampler)
    PointEnv.__init__(self,
                      file_path=file_path,
                      expose_all_qpos=expose_all_qpos,
                      non_zero_reset=non_zero_reset)

class GoalReachingPointDictEnv(goal_reaching_env.GoalReachingEnv, PointEnv):
  """Ant locomotion for goal reaching in a disctionary compatible format."""
  BASE_ENV = PointEnv

  def __init__(self, goal_sampler=goal_reaching_env.disk_goal_sampler,
               file_path=None,
               expose_all_qpos=False):
    goal_reaching_env.GoalReachingEnv.__init__(self, goal_sampler)
    PointEnv.__init__(self, 
                    file_path=file_path,
                    expose_all_qpos=expose_all_qpos)

class PointMazeEnv(maze_env.MazeEnv, GoalReachingPointEnv):
  """Point navigating a maze."""
  LOCOMOTION_ENV = GoalReachingPointEnv

  def __init__(self, goal_sampler=None, expose_all_qpos=True,
               *args, **kwargs):
    if goal_sampler is None:
      goal_sampler = lambda np_rand: maze_env.MazeEnv.goal_sampler(self, np_rand)
    maze_env.MazeEnv.__init__(
        self, *args, manual_collision=True,
        goal_sampler=goal_sampler,
        expose_all_qpos=expose_all_qpos,
        **kwargs)

def make_point_maze_env(**kwargs):
  env = PointMazeEnv(**kwargs)
  # print(env.action_space)
  return wrappers.NormalizedBoxEnv(env)

def create_goal_reaching_policy(obs_to_goal=lambda obs: obs[-2:],
                                obs_to_ori=lambda obs: obs[0]):
  """A hard-coded policy for reaching a goal position."""

  def policy_fn(obs):
    goal_x, goal_y = obs_to_goal(obs)
    goal_dist = np.linalg.norm([goal_x, goal_y])
    goal_ori = np.arctan2(goal_y, goal_x)
    ori = obs_to_ori(obs)
    ori_diff = (goal_ori - ori) % (2 * np.pi)

    radius = goal_dist / 2. / max(0.1, np.abs(np.sin(ori_diff)))
    rotation_left = (2 * ori_diff) % np.pi
    circumference_left = max(goal_dist, radius * rotation_left)

    speed = min(circumference_left * 5., 1.0)
    velocity = speed
    if ori_diff > np.pi / 2 and ori_diff < 3 * np.pi / 2:
      velocity *= -1

    time_left = min(circumference_left / (speed * 0.2), 10.)
    signed_ori_diff = ori_diff
    if signed_ori_diff >= 3 * np.pi / 2:
      signed_ori_diff = 2 * np.pi - signed_ori_diff
    elif signed_ori_diff > np.pi / 2 and signed_ori_diff < 3 * np.pi / 2:
      signed_ori_diff = signed_ori_diff - np.pi

    angular_velocity = signed_ori_diff / time_left
    angular_velocity = np.clip(angular_velocity, -1., 1.)

    return np.array([velocity, angular_velocity])

  return policy_fn


def create_maze_navigation_policy(maze_env):
  """Creates a hard-coded policy to navigate a maze."""
  ori_index = 2 if maze_env._expose_all_qpos else 0
  obs_to_ori = lambda obs: obs[ori_index]

  goal_reaching_policy = create_goal_reaching_policy(obs_to_ori=obs_to_ori)
  goal_reaching_policy_fn = lambda obs, goal: goal_reaching_policy(
    np.concatenate([obs, goal]))

  return maze_env.create_navigation_policy(goal_reaching_policy_fn)
