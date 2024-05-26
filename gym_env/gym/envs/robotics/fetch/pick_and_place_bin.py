import os
from gym import utils
from gym.envs.robotics import fetch_env_bin


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place_bin.xml')


class FetchPickAndPlaceBinEnv(fetch_env_bin.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            # 'solid1wall': [1.27, 0.65, 0.4, 1., 0., 0., 1.],
            # 'solid2wall': [1.075, 0.85, 0.4, 1., 0., 0., 1.],
            # 'hollow1wall': [1.07, 0.75, 0.4, 1., 0., 0., 1.],
            # 'hollow2wall': [1.35, 0.75, 0.4, 1., 0., 0., 1.],
            # 'hollow3wall': [1.3, 0.715, 0.4, 1., 0., 0., 0.],
            # 'hollow4wall': [1.3, 1.01, 0.4, 1., 0., 0., 0.],
            # 'door1': [1.24, 0.65, 0.4, 1., 0., 0., 0.],
            # 'door2': [1.3, 0.65, 0.4, 1., 0., 0., 0.],
        }
        fetch_env_bin.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.2, target_range=0.2, distance_threshold=0.1,
            initial_qpos=initial_qpos, reward_type=reward_type, image_obs=True, randomize = False, fixed_goal=False)
        utils.EzPickle.__init__(self)
