from gym import utils
from gym.envs.robotics import fetch_multi_goal_env


class FetchMultiGoalReachEnv(fetch_multi_goal_env.FetchMultiGoalEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', num_goals=2):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        fetch_multi_goal_env.FetchMultiGoalEnv.__init__(
            self, 'fetch/reach_multi_goal.xml', has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.02,
            initial_qpos=initial_qpos, reward_type=reward_type, num_goals=num_goals)
        utils.EzPickle.__init__(self)
