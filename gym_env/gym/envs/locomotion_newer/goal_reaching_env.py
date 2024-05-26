import numpy as np


def disk_goal_sampler(np_random, goal_region_radius=10., index = -1):
  th = 2 * np.pi * np_random.uniform()
  radius = goal_region_radius * np_random.uniform()
  return radius * np.array([np.cos(th), np.sin(th)])

def constant_goal_sampler(np_random, location=10.0 * np.ones([2])):
  return location

class GoalReachingEnv(object):
  """General goal-reaching environment."""
  BASE_ENV = None  # Must be specified by child class.

  def __init__(self, goal_sampler, eval=False, reward_type='sparse'):
    self._goal_sampler = goal_sampler
    self._goal = np.ones([3])
    # self._maze_array = None
    self.target_goal = self._goal
    self.distance_threshold = 1.5

    # This flag is used to make sure that when using this environment
    # for evaluation, that is no goals are appended to the state
    self.eval = eval

    # This is the reward type fed as input to the goal confitioned policy
    self.reward_type = reward_type

  def _get_obs(self):
    base_obs = self.BASE_ENV._get_obs(self).copy()
    object_pos_temp = np.array([base_obs[0], base_obs[1], 0.0])
    maze_array = self._maze_array
    # obs_temp = np.concatenate([base_obs.copy(), maze_array.copy().ravel()])
    obs_temp = np.concatenate([base_obs.copy()])
    obs = {'observation': obs_temp.copy(),
          'achieved_goal': object_pos_temp.copy(),
          'desired_goal': np.array(self.target_goal.copy())
    }
    print(obs)
    # goal_direction = self._goal - self.get_xy()
    # if not self.eval:
    #   obs = np.concatenate([base_obs, goal_direction])
    #   return obs
    # else:
    return obs

  def step(self, a):
    _, _, _, info = self.BASE_ENV.step(self, a)
    object_pos = self.get_xy()
    object_pos = np.array([[object_pos[0], object_pos[1], 1.0]])
    reward = self.compute_reward(object_pos , np.array([self.target_goal]), reward_type=self.reward_type)
    # reward = self.get_reward()
    done = False
    # Terminate episode when we reach a goal
    # if self.eval and np.linalg.norm(object_pos - self.target_goal) <= self.distance_threshold:
    #   done = True

    obs = self._get_obs().copy()
    return obs, reward, done, info

  # def compute_reward(self, achieved_goal, goal, info=None, reward_type='sparse'):
  #   # Compute distance between goal and the achieved goal.
  #   # d = goal_distance(achieved_goal, goal)
  #   reward = 0.0 if np.linalg.norm(achieved_goal - goal, axis=-1) <= self.distance_threshold else -1.0
  #   return reward
  #   # if reward_type == 'sparse':
  #   #     # print(-(d > self.distance_threshold).astype(np.float32))
  #   #     return -(d > self.distance_threshold).astype(np.float32)
  #   # else:
  #   #     return -d

  def compute_reward(self, achieved_goal, goal, info=None, reward_type='sparse'):
        reward_type='dense'
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        if reward_type == 'sparse':
            # print(-(d > self.distance_threshold).astype(np.float32))
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

  def goal_distance(self, goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

  def _reset_model(self):
    # if self.target_goal is not None or self.eval:
    #   self._goal = self.target_goal
    # else:
    #   self._goal = self._goal_sampler(self.np_random)
    return self.BASE_ENV._reset_model(self)
