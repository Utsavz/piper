import os
import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding
import cv2
import time

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


class RobotEnv(gym.GoalEnv):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None

        self.max_u = [0.25, 0.27, 0.145]
        self.action_offset = [1.3, 0.75, 0.555]

        self.last_action = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        # self._reset_sim()
        self.goal = self._sample_goal()
        self.flag = 1
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

        self.render_permit = 0

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step_maze(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def step_pick(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def step_rope(self, action):
        x_min = 1.05
        x_max = 1.55
        y_min = 0.48
        y_max = 1.02
        self.collision_with_initial_rope = 0
        action = np.clip(action, self.action_space.low, self.action_space.high)
        drag_len = 0.08
        action = action.copy()  # ensure that we don't change the action outside of this scope
        ac_start = np.array([action[0], action[1], 0.5])
        # ac_end = np.array([action[2], action[3], 0.5])
        drag_theta = action[2]*180+180.0
        # Bring in 0 to 1 range
        action[3] = (action[3]+1)/2.0
        # drag_len = action[3]*drag_len
        bins = 18
        # drag_theta = drag_theta - drag_theta % 20.0
        action_start = self.action_offset + (self.max_u * ac_start)
        # action_end = self.action_offset + (self.max_u * ac_end)

        action_x = action_start[0] + drag_len*np.cos(np.radians(drag_theta))*1.0
        action_y = action_start[1] + drag_len*np.sin(np.radians(drag_theta))*1.0
        action_x = max(x_min, action_x)
        action_x = min(x_max, action_x)
        action_y = max(y_min, action_y)
        action_y = min(y_max, action_y)
        action_end = np.array([action_x, action_y, 0.5])
        action_start[2] = 0.5
        action_end[2] = 0.5
        # action_start = [1.3, 0.65, 0.5]
        # action_end = [1.3, 0.95, 0.5]
        action_start_temp = action_start.copy()
        action_end_temp = action_end.copy()
        action_start_temp[2] = 0.42
        action_end_temp[2] = 0.42
        # self.set_subgoal('subgoal_2', action_start_temp)
        # self.set_subgoal('subgoal_0', action_end_temp)

        grip_pos = self.sim.data.get_site_xpos('robot0:grip').copy()
        object_oriented_goal = action_start - grip_pos
        initial_grip_pos = self.sim.data.get_site_xpos('robot0:grip').copy()
        initial_action_start = initial_grip_pos.copy()
        initial_action_start[2] = 0.6
        initial_object_oriented_goal = initial_action_start - initial_grip_pos
        
        image_count = 0
        count = 0

        while np.linalg.norm(initial_object_oriented_goal) >= 0.03:
            if count > 20:
                break
            else:
                count += 1
            
            # image = self.capture()
            image_count += 1
            
            # cv2.imwrite('/home/vrsystem/gitrep/hachathon/to_del_images/test_image'+str(image_count)+'.png', image)
            action = np.array([0., 0., 0., 0.])
            action[0] = initial_object_oriented_goal[0]*6
            action[1] = initial_object_oriented_goal[1]*6
            action[2] = initial_object_oriented_goal[2]*6
            action[3] = 0

            action_temp = action.copy()
            # print('action_temp', action_temp)
            self._set_action(action_temp)
            self.sim.step()
            self._step_callback()

            initial_grip_pos = self.sim.data.get_site_xpos('robot0:grip')
            # print(initial_grip_pos)
            initial_object_oriented_goal = initial_action_start - initial_grip_pos

        grip_pos = self.sim.data.get_site_xpos('robot0:grip').copy()
        object_oriented_goal = action_start - grip_pos

        while np.linalg.norm(object_oriented_goal) >= 0.03:
            if count > 20:
                break
            else:
                count += 1
            if self.render_permit:
                self.render()
            # image = self.capture()
            image_count += 1
            
            # cv2.imwrite('/home/vrsystem/gitrep/hachathon/to_del_images/test_image'+str(image_count)+'.png', image)
            action = np.array([0., 0., 0., 0.])
            action[0] = object_oriented_goal[0]*6
            action[1] = object_oriented_goal[1]*6
            action[2] = object_oriented_goal[2]*6
            action[3] = 0

            action_temp = action.copy()
            # print('action_temp', action_temp)
            self._set_action(action_temp)
            self.sim.step()
            self._step_callback()

            grip_pos = self.sim.data.get_site_xpos('robot0:grip')
            object_oriented_goal = action_start - grip_pos
            # print(action_start)
            # print(grip_pos)
            # print(object_oriented_goal)
            # print(np.linalg.norm(object_oriented_goal))
            # print('1')

        rope_size = 15
        # for num in range(rope_size):
        #     curr_pos = self.sim.data.get_geom_xpos('G'+str(num))
        #     curr_dist = np.linalg.norm(grip_pos[:2] - curr_pos[:2])
        #     if curr_dist < 0.03:
        #         curr_pos1 = self.sim.data.get_geom_xpos('G'+str(max(0,num-1)))
        #         curr_pos2 = self.sim.data.get_geom_xpos('G'+str(min(14,num+1)))
        #         # print('curr_pos', curr_pos1, curr_pos2)
        #         x1, y1 = ( curr_pos1[1], -curr_pos1[0])
        #         x2, y2 = ( curr_pos2[1], -curr_pos2[0])

        #         # x:[1.05, 1.55]
        #         # y:[0.48, 1.02]
        #         slope = (y2-y1) / (x2-x1+0.000001)
        #         # print('here before', action_start, slope)
        #         drag_sign = 1
        #         # if np.random.rand() < 0.5:
        #         #     drag_sign = -1
        #         action_start[0] = action_start[0] + drag_sign*0.06*np.cos(np.arctan(slope))
        #         action_start[1] = action_start[1] + drag_sign*0.06*np.sin(np.arctan(slope))
        #         action_start[0] = max(x_min, action_start[0])
        #         action_start[0] = min(x_max, action_start[0])
        #         action_start[1] = max(y_min, action_start[1])
        #         action_start[1] = min(y_max, action_start[1])

        #         for i in range(5):
        #             iteration = 0
        #             for num in range(rope_size):
        #                 iteration += 1
        #                 curr_pos = self.sim.data.get_geom_xpos('G'+str(num))
        #                 curr_dist = np.linalg.norm(action_start[:2] - curr_pos[:2])
        #                 if curr_dist < 0.03:
        #                     action_start[0] = action_start[0] + drag_sign*0.06*np.cos(np.arctan(slope))
        #                     action_start[1] = action_start[1] + drag_sign*0.06*np.sin(np.arctan(slope))
        #                     action_start[0] = max(1.05, action_start[0])
        #                     action_start[0] = min(1.55, action_start[0])
        #                     action_start[1] = max(0.48, action_start[1])
        #                     action_start[1] = min(1.02, action_start[1])
        #             if iteration == rope_size:
        #                 break
        #         # print('here', action_start, iteration)
        #         # self.set_subgoal('subgoal_0', action_start)
        #         # self.render()
        #         # image = self.capture()
        #         image_count += 1
                
        #         # cv2.imwrite('/home/vrsystem/gitrep/hachathon/to_del_images/test_image'+str(image_count)+'.png', image)
        #         count = 0
        #         while np.linalg.norm(object_oriented_goal) >= 0.03:
        #             if count > 20:
        #                 break
        #             else:
        #                 count += 1
        #             # self.render()
        #             # image = self.capture()
        #             image_count += 1
                    
        #             # cv2.imwrite('/home/vrsystem/gitrep/hachathon/to_del_images/test_image'+str(image_count)+'.png', image)
        #             action = np.array([0., 0., 0., 0.])
        #             action[0] = object_oriented_goal[0]*6
        #             action[1] = object_oriented_goal[1]*6
        #             action[2] = object_oriented_goal[2]*6
        #             action[3] = 0

        #             action_temp = action.copy()
        #             # print('action_temp', action_temp)
        #             self._set_action(action_temp)
        #             self.sim.step()
        #             self._step_callback()

        #             grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        #             object_oriented_goal = action_start - grip_pos
        #         break
                # assert False


        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        object_oriented_goal = action_start - grip_pos
        
        image_count = 0
        count = 0
        while np.linalg.norm(object_oriented_goal) >= 0.03:
            if count > 20:
                break
            else:
                count += 1
            if self.render_permit:
                self.render()
            # image = self.capture()
            image_count += 1
            
            # cv2.imwrite('/home/vrsystem/gitrep/hachathon/to_del_images/test_image'+str(image_count)+'.png', image)
            action = np.array([0., 0., 0., 0.])
            action[0] = object_oriented_goal[0]*6
            action[1] = object_oriented_goal[1]*6
            action[2] = object_oriented_goal[2]*6
            action[3] = 0

            action_temp = action.copy()
            # print('action_temp', action_temp)
            self._set_action(action_temp)
            self.sim.step()
            self._step_callback()

            grip_pos = self.sim.data.get_site_xpos('robot0:grip')
            object_oriented_goal = action_start - grip_pos


        action_start[2] = 0.25
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        object_oriented_goal = action_start - grip_pos

        # print(np.linalg.norm(object_oriented_goal))
        # assert False
        count = 0
        while np.linalg.norm(object_oriented_goal) >= 0.03:
            if count > 20:
                break
            else:
                count += 1
            if self.render_permit:
                self.render()
            if self.if_collision():
            	self.collision_with_initial_rope = 1
            # image = self.capture()
            image_count += 1
            
            # cv2.imwrite('/home/vrsystem/gitrep/hachathon/to_del_images/test_image'+str(image_count)+'.png', image)
            action = np.array([0., 0., 0., 0.])
            action[0] = object_oriented_goal[0]*6
            action[1] = object_oriented_goal[1]*6
            action[2] = object_oriented_goal[2]*6
            action[3] = 0

            action_temp = action.copy()
            # print('action_temp', action_temp)
            self._set_action(action_temp)
            self.sim.step()
            self._step_callback()

            grip_pos = self.sim.data.get_site_xpos('robot0:grip')
            object_oriented_goal = action_start - grip_pos
            # print(object_oriented_goal)
            # print(np.linalg.norm(object_oriented_goal))
            # print('2')

        action_start[:2] = action_end[:2].copy()
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        object_oriented_goal = action_start - grip_pos

        count = 0
        while np.linalg.norm(object_oriented_goal) >= 0.04:
            if count > 20:
                break
            else:
                count += 1
            if self.render_permit:
                self.render()
            if self.if_collision():
            	self.collision_with_initial_rope = 1
            # image = self.capture()
            image_count += 1
            
            # cv2.imwrite('/home/vrsystem/gitrep/hachathon/to_del_images/test_image'+str(image_count)+'.png', image)
            action = np.array([0., 0., 0., 0.])
            action[0] = object_oriented_goal[0]*6
            action[1] = object_oriented_goal[1]*6
            action[2] = object_oriented_goal[2]*6
            # print('oo',object_oriented_goal)
            # import time
            # time.sleep(.5)
            action[3] = 0

            action_temp = action.copy()
            # print('action_temp', action_temp)
            self._set_action(action_temp)
            self.sim.step()
            self._step_callback()

            grip_pos = self.sim.data.get_site_xpos('robot0:grip')
            object_oriented_goal = action_start - grip_pos
            # print(grip_pos)
            # print(action_start)
            # print(object_oriented_goal)
            # print(np.linalg.norm(object_oriented_goal))
            # print('3')

        action_start[2] = 0.5
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        object_oriented_goal = action_start - grip_pos

        count = 0
        while np.linalg.norm(object_oriented_goal) >= 0.01:
            if count > 20:
                break
            else:
                count += 1
            if self.render_permit:
                self.render()
            if self.if_collision():
            	self.collision_with_initial_rope = 1
            # image = self.capture()
            image_count += 1
            
            # cv2.imwrite('/home/vrsystem/gitrep/hachathon/to_del_images/test_image'+str(image_count)+'.png', image)
            action = np.array([0., 0., 0., 0.])
            action[0] = object_oriented_goal[0]*6
            action[1] = object_oriented_goal[1]*6
            action[2] = object_oriented_goal[2]*6
            action[3] = 0

            action_temp = action.copy()
            # print('action_temp', action_temp)
            self._set_action(action_temp)
            self.sim.step()
            self._step_callback()

            grip_pos = self.sim.data.get_site_xpos('robot0:grip')
            object_oriented_goal = action_start - grip_pos
            # print(object_oriented_goal)
            # print(np.linalg.norm(object_oriented_goal))
            # print('3')



        obs = self._get_obs()
        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None

    def render(self, mode='human'):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer().render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            data = self._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()
        # import cv2
        # self.flag += 1
        # image = self.capture()
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)[:,:]
        # image = image[200:1000,500:1450]
        # self.flag += 1
        # cv2.imwrite('./images/rope/rope_'+str(self.flag).zfill(4)+'.png', image)

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self._viewer_setup()
        return self.viewer


    def set_subgoal(self, name, action):
        # site_id = self.sim.model.site_name2id(name)
        # self.sim.model.site_pos[site_id] = action
        raise NotImplementedError()


    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def if_collision(self):
        """Collision detection.
        """
        pass

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
