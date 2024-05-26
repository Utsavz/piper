import numpy as np
from mujoco_py.generated import const
import cv2
from gym.envs.robotics import rotations, robot_env, utils
from shapely.geometry import Polygon, Point, MultiPoint
import random
from mpi4py import MPI

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

    # if goal_a.ndim == 1:
    #     return np.linalg.norm(goal_a - goal_b, axis=-1)
    # else:
    #     return np.zeros(goal_a.shape[0])
    # else:
    #     curr_list = []
    #     curr_max = 0.
    #     for j in range(goal_a.shape[0]):
    #         for i in range(int(len(goal_a[j])/3)):
    #             current_norm = np.linalg.norm(goal_a[j][3*i:3*i+3] - goal_b[j][3*i:3*i+3], axis=-1)
    #             if current_norm > curr_max:
    #                 curr_max = current_norm
    #         curr_list.append(curr_max)
    #     return np.array(curr_list)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,image_obs=None, random_maze=False,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.image_obs = 1
        self.count=0
        self.block_gripper = True
        self.target_in_the_air = False
        self.randomize = 0
        self.random_maze = random_maze
        self.index = 0
        self.flag = 0
        self.maze_array = None
        self.collision_with_initial_rope = 0
        self.fixed_goal = 1
        self.generate_random_maze = 0
        self.generate_three_room_maze = 0
        self.generate_four_room_maze = 0
        self.generate_five_room_maze = 0
        self.generate_six_room_maze = 0
        self.generate_eight_room_maze = 0
        self.generate_one_wall_maze = 0
        self.generate_two_wall_maze = 0
        curr_eps_num_wall_collisions = []
        self.rank_seed =  np.random.get_state()[1][0]
        # print('Data Loading...')
        data = np.load('/home/vrsystem/gitrep/hachathon/clutter_data_generation/robotic_rope_dataset_5_action_3_5000_simplified.npz')
        # data = np.load('/home/vrsystem/gitrep/hachathon/clutter_data_generation/robotic_rope_dataset_5_action_3_5000_test_unseen_simplified.npz')
        # data = np.load('/home/vrsystem/gitrep/hachathon/clutter_data_generation/robotic_rope_dataset_5_action_3_100_test_unseen_0.05_simplified.npz')
        # data = np.load('/home/vrsystem/gitrep/hachathon/clutter_data_generation/robotic_rope_dataset_5_action_3_100_test_unseen_0.1_simplified.npz')
        # data = np.load('/home/vrsystem/gitrep/hachathon/clutter_data_generation/robotic_rope_dataset_5_action_3_100_demos_linear_interpolation.npz')
        # data = np.load('/home/vrsystem/gitrep/hachathon/clutter_data_generation/robotic_rope_lower_dataset_x1_x2_action_angle_5_50000_simplified.npz')
        # print('Loading data for env')
        self.obs = data['obs'].copy()
        # self.acs = data['acs'][:5000].copy()
        self.states = data['states'].copy()
        del data
        # tempp = {'observation': np.zeros(45),'achieved_goal':np.zeros(45), 'desired_goal':np.zeros(45) }
        # obs_temp = []
        # for i in range(10000):
        #     obs_temp.append(tempp)
        # obs = []
        # for i in range(10):
        #     obs.append(obs_temp)
        # self.obs = np.array(obs)
        # self.acs = np.zeros((10000,10,4))
        # self.states = np.zeros((10000,10,100))
        # Hor_left, Hor_right, Vert_up, vert_down coordinates
        self.gates = [-1 for _ in range(8)]

        # if "Pick" in self.__class__.__name__:
        # self.polygons_list=[Polygon([(1.06, 0.73),(1.12,0.73),(1.12,0.77),(1.06, 0.77)]),#verti_up   
        #                     Polygon([(1.215, 0.73),(1.38,0.73),(1.38,0.77),(1.215, 0.77)]),#verti_mid  
        #                     Polygon([(1.48, 0.73),(1.54,0.73),(1.54,0.77),(1.48, 0.77)]),#verti_down 
        #                     Polygon([(1.28, 0.48),(1.32,0.48),(1.32, 0.57),(1.28, 0.57)]),#hori_left  
        #                     Polygon([(1.28, 0.668),(1.32,0.668),(1.32, 0.835),(1.28, 0.835)]),#hori_mid   
        #                     Polygon([(1.28, 0.933),(1.32,0.933),(1.32, 1.02),(1.28, 1.02)]),#hori_right
        #                     Polygon([(1.06, 0.63),(1.35,0.63),(1.35, 0.67),(1.06, 0.67)]),# up_left maze
        #                     Polygon([(1.26, 0.83),(1.54,0.83),(1.54, 0.87),(1.26, 0.87)]),# down_right maze
        #                     Polygon([(1.26, 0.63),(1.54,0.63),(1.54, 0.67),(1.26, 0.67)]),# down_left maze
        #                     Polygon([(1.06, 0.83),(1.35,0.83),(1.35, 0.87),(1.06, 0.87)]),# up_right maze
        #                     Polygon([(1.28, 0.57),(1.32,0.57),(1.32, 0.668),(1.28, 0.668)]),#hori_mid_left_door
        #                     Polygon([(1.28, 0.835),(1.32,0.835),(1.32, 0.933),(1.28, 0.933)]),#hori_mid_right_door
        #                     Polygon([(1.12, 0.73),(1.215,0.73),(1.215,0.77),(1.12, 0.77)]),#verti_mid_up_door 
        #                     Polygon([(1.38, 0.73),(1.48,0.73),(1.48,0.77),(1.38, 0.77)])]#verti_mid_down_door 

        # self.polygons = []
        # self.obstacle_list=[(0,0,0,0,0,0,0,0,0,0,0,0,0,0),
        #                     (0,0,0,0,0,0,1,1,0,0,0,0,0,0),
        #                     (0,0,0,0,0,0,0,0,1,1,0,0,0,0),
        #                     (1,1,1,1,1,1,0,0,0,0,0,0,0,0),
        #                     (1,1,1,0,0,0,0,0,0,0,0,0,0,0),
        #                     (0,0,0,1,1,1,0,0,0,0,0,0,0,0),
        #                     (1,1,0,0,1,1,0,0,0,0,0,0,0,0),
        #                     (0,1,1,0,1,1,0,0,0,0,0,0,0,0),
        #                     (0,1,1,1,1,0,0,0,0,0,0,0,0,0),
        #                     (1,1,0,1,1,0,0,0,0,0,0,0,0,0),
        #                     (1,1,1,0,1,1,0,0,0,0,0,0,0,0),
        #                     (1,1,1,1,1,0,0,0,0,0,0,0,0,0),
        #                     (0,1,1,1,1,1,0,0,0,0,0,0,0,0),
        #                     (1,1,0,1,1,1,0,0,0,0,0,0,0,0),
        #                     (1,1,1,0,0,0,0,0,0,0,0,0,1,0),
        #                     (1,1,1,0,0,0,0,0,0,0,0,0,0,1),
        #                     (1,1,1,0,1,1,0,0,0,0,0,0,1,0),
        #                     (1,1,0,1,1,1,0,0,0,0,1,0,0,0),
        #                     (1,1,1,1,1,0,0,0,0,0,0,0,1,0),
        #                     (0,1,1,1,1,1,0,0,0,0,1,0,0,0)]

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

        self.body_pos_initial_backup = self.sim.model.body_pos.copy()

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info=None, reward_type='sparse'):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        # return d
        # print(-d)
        if reward_type == 'sparse':
            rew = -(d > self.distance_threshold).astype(np.float32)
            # print(rew)
            return rew
        else:
            return -d

    def set_subgoal(self, name, action):
        site_id = self.sim.model.site_name2id(name)
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        self.sim.model.site_pos[site_id] = action - sites_offset[0]
        self.sim.model.site_rgba[site_id][3] = 1

    def reset_subgoal(self, name, action):
        site_id = self.sim.model.site_name2id(name)
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        self.sim.model.site_pos[site_id] = action - sites_offset[0]
        self.sim.model.site_rgba[site_id][3] = 0


    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action_rope(self.sim, action)

    def _get_obs(self):
        # positions
        # grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        # dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        # grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        # robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        # if self.has_object:
        #     object_pos = self.sim.data.get_site_xpos('object0')
        #     # rotations
        #     object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
        #     # velocities
        #     object_velp = self.sim.data.get_site_xvelp('object0') * dt
        #     object_velr = self.sim.data.get_site_xvelr('object0') * dt
        #     # gripper state
        #     object_rel_pos = object_pos - grip_pos
        #     object_rel_pos = object_rel_pos.copy()
        #     object_velp -= grip_velp
        #     object_velp = object_velp.copy()
        # else:
        #     object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        # gripper_state = robot_qpos[-2:]
        # gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        # if not self.has_object:
        #     achieved_goal = grip_pos.copy()
        # else:
        #     achieved_goal = np.squeeze(object_pos.copy())

        # if self.image_obs:
        #     image = self.get_image_obs(grip_pos, object_pos)

        # if self.maze_array is not None:
        #     maze_pos = self.maze_array.copy()
        #     maze = maze_pos.ravel()
        # else:
        #     maze = [0]*121

        # maze = (np.array(maze).astype(np.uint8)).tolist()

        # maze = [(0,0)]*122    
        # flag = 0
        # if self.maze_array is not None:
        #     for i in range(self.maze_array.shape[0]):
        #         for j in range(self.maze_array.shape[1]):
        #             flag += 1
        #             if self.maze_array[i][j]:
        #                 maze[flag] = (5*(i+1)/2.0, 4.6*(j+1)/2.0) #imgWall[5*i:5*(i+1),int(4.6*(j)):int(4.6*(j+1))] = 255
        # maze = np.array(maze).ravel()


        # obs = np.concatenate([
        #     grip_pos, object_pos.ravel(), object_rel_pos.ravel(), maze, #image.ravel(), object_rot.ravel(),
        #     # object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        #     ])

        rope_size = 15
        rope_pos = np.zeros((rope_size, 2))
        for num in range(rope_size):
            curr_pos = self.sim.data.get_geom_xpos('G'+str(num))[:2].copy()
            rope_pos[num] = curr_pos
        rope_pos = rope_pos.ravel().copy()

        # rope_pos = self.sim.data.get_geom_xpos('G14')
        # for i in range(4,19):
        #     self.set_subgoal('subgoal_'+str(i), self.sim.data.get_geom_xpos('G'+str(i-4)))
        # print(rope_pos)

        if not self.image_obs:
            obs = np.concatenate([
                grip_pos, object_pos.ravel(), object_rel_pos.ravel(), maze#np.array([self.getIndex()]),#maze, #self.gates.copy(), #object_rot.ravel(),
                #object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
                ])
        else:
            obs = np.concatenate([
                rope_pos.ravel(),#image,#maze, #self.gates.copy(), #object_rot.ravel(),
                #object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
                ])

        # import cv2
        # self._get_viewer()
        # self.render()
        # self.flag += 1
        # image = self.capture()
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)[:,:]
        # image = image[200:1000,500:1450]
        # # cv2.imshow('test', image)
        # print(self.flag)
        # cv2.imwrite('./images/rope/rope_'+str(self.flag)+'.png', image)


        # obs = np.concatenate([grip_pos, object_pos.ravel(), object_rel_pos.ravel(), self.obstacle_list[self.index]])

        # cv2.imwrite('./images/tester'+str(self.count)+'.png', image)#cv2.cvtColor(imgGri, cv2.COLOR_RGB2BGR)[:,:])
        # self.count += 1
        return {
            'observation': obs.copy(),
            'achieved_goal': rope_pos.copy(),
            'desired_goal': self.goal.copy(),
        }

    # def map_cor(self, pos,X1=0.0,X2=273.0,Y1=0.,Y2=467.0,y2=1.06,y1=0.44,x1=1.05,x2=1.53):
    #     # if pos[0] > 1:
    #     #     return np.array([193., 180.])
    #     x = pos[0]
    #     y = pos[1]
    #     X = X1 + ( (x-x1) * (X2-X1) / (x2-x1) )
    #     Y = Y1 + ( (y-y1) * (Y2-Y1) / (y2-y1) )
    #     # X=X2+((X1-X2)/x2)*x
    #     # Y=Y2+((Y1-Y2)/y2)*y
    #     return(np.array([X,Y]))

    def map_cor(self, pos, X1=0.0,X2=30.0,Y1=0.,Y2=30.0,x1=1.05,x2=1.55,y1=0.48,y2=1.02):
        # if pos[0] > 1:
        #     return np.array([193., 180.])
        x = pos[0]
        y = pos[1]
        # if x<1.0 or x>1.6 or y<0.43 or y>1.07:
        #     return np.array([290,480])

        X = X1 + ( (x-x1) * (X2-X1) / (x2-x1) )
        Y = Y1 + ( (y-y1) * (Y2-Y1) / (y2-y1) )
        return(np.array([X,Y]))

    def get_image_obs(self, grip_pos, object_pos):
        import cv2
        blackBox = grip_pos.copy()
        redCircle = self.goal.copy()
        blueBox = object_pos.copy()
        height = 30
        width = 30
        imgGripper = np.zeros((height,width), np.uint8)
        imgWall = np.zeros((height,width), np.uint8)
        imgGoal = np.zeros((height,width), np.uint8)
        # half_block_len = int(38/2)
        # gripper_len = 15
        # sphere_rad = 18
        half_block_len = 1.2
        gripper_len = 1.2
        sphere_rad = 1.2
        # Mark the block position
        # if len(blueBox) != 0:
        #     mapBlue = self.map_cor(blueBox)
        #     xx = int(mapBlue[0])
        #     yy = int(mapBlue[1])
        #     cv2.circle(imgBlock,(yy,xx), sphere_rad, (255), -1)
            # imgBlock[max(0,xx-half_block_len):xx+half_block_len,max(0,yy-half_block_len):yy+half_block_len] = 255

        # flag = 0
        if self.maze_array is not None:
            for i in range(self.maze_array.shape[0]):
                for j in range(self.maze_array.shape[1]):
                    # flag += 1
                    if self.maze_array[i][j]:
                        imgWall[int(3*i):int(3*(i+1)),int(2.76*(j)):int(2.76*(j+1))] = 255 #maze[flag] = (5*(i+1)/2.0, 4.6*(j+1)/2.0)

        # Mark the sphere position
        # mapRed = self.map_cor(redCircle)
        # xx = int(mapRed[0])
        # yy = int(mapRed[1])
        # cv2.circle(imgGoal,(yy,xx), sphere_rad, (255), -1)

        # # Mark the gripper position
        # mapBlack = self.map_cor(blackBox)
        # xx = int(mapBlack[0])
        # yy = int(mapBlack[1])
        # cv2.circle(imgGripper,(yy,xx), sphere_rad, (255), -1)
        # imgGripper[max(0,xx-gripper_len):xx+gripper_len,max(0,yy-gripper_len):yy+gripper_len] = (255)
        # image = np.dstack((imgGripper, imgGoal, imgWall))
        image = imgWall

        # image = cv2.resize(image, (50,50))
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)[:,:]
        image = image/255.
        image = image.astype(np.uint8)
        # assert False
        # cv2.imshow('fet_env',image)
        # cv2.waitKey(0)
        obs = image.ravel().copy()
        return obs

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('table0')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.fixedcamid = 3
        self.viewer.cam.type = const.CAMERA_FIXED
        self.viewer.cam.distance = 1.8
        self.viewer.cam.azimuth = 140.
        self.viewer.cam.elevation = -30.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        if self.goal.shape[0] == 3:
            self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self, seed=0):
        self.sim.set_state(self.initial_state)
        # Randomize the environment
        if self.randomize:
            if self.random_maze:
                self.randomize_environ(seed)
            else:
                # Randomize the walls to make it a dynamic environment
                self.randomize_environ_room_env()

        self.curr_eps_num_wall_collisions = []
        if self.has_object:
            object_xpos = self.get_object_pos()
            while self.check_overlap(object_xpos):
                object_xpos = self.get_object_pos()
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def setIndex(self, index):
        self.index = index
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim(index)
        # Set initial state
        env_state_current = self.sim.get_state().flatten().copy()
        curr_states_list = self.states#[index]
        random_pick = np.random.randint(len(curr_states_list))
        random_pick_1 = np.random.randint(len(curr_states_list[random_pick]))
        env_state_old = curr_states_list[random_pick][random_pick_1].copy()
        env_state_old[0] = env_state_current[0]
        self.sim.set_state_from_flattened(env_state_old)

        # Move end effector into position.
        gripper_target = np.array([1.13, 0.65, 0.5])
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        self.sim.step()


        # Set goal state
        self.goal = self._sample_goal(index).copy()
        self.curr_eps_num_wall_collisions = []
        obs = self._get_obs()
        return obs

    def getIndex(self):
        return self.index

    def get_object_pos(self):
        object_xpos = self.initial_gripper_xpos[:2]
        while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.05:
            object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            # object_xpos = [1.2, .75]
        return object_xpos.copy()


    def _set_state(self, state, goal):
        env_state_current = self.sim.get_state().flatten().copy()
        env_state_old = state.copy()
        env_state_old[0] = env_state_current[0]
        self.sim.set_state_from_flattened(env_state_old)
        self.goal = goal

        # Simulation step
        for _ in range(10):
            self.sim.step()
            self.sim.forward()
        return True

    def _get_state(self):
        return self.sim.get_state().flatten().copy()

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
        elif self.generate_one_wall_maze:
            ret_array = self.generate_maze_one_wall(width=width, height=height, complexity=complexity, density=density, seed=seed)
        elif self.generate_two_wall_maze:
            ret_array = self.generate_maze_two_wall(width=width, height=height, complexity=complexity, density=density, seed=seed)
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

    # def randomize_environ(self, seed=0):
    #     # Prim's maze generation algorithm
    #     # self.index = 0
    #     # seed = self.index
    #     maze_array = self.generate_maze(seed=seed)
    #     self.maze_array = maze_array.copy()
    #     num = 0
    #     for col in range(10):
    #         for row in range(11):
    #             num += 1
    #             if maze_array[col][row] == 1:
    #                 name = "object"+str(num)+":joint"
    #                 object_qpos = self.sim.data.get_joint_qpos(name)
    #                 object_qpos[0] = 1.08
    #                 object_qpos[1] = 0.5
    #                 object_qpos[0] += col*0.049
    #                 object_qpos[1] += row*0.05
    #                 object_qpos[2] = 0.42
    #                 self.sim.data.set_joint_qpos(name, object_qpos)

    def randomize_environ(self, seed=0):
        # Prim's maze generation algorithm
        # self.index = 0
        # seed = self.index
        maze_array = self.generate_maze(seed=seed)
        self.maze_array = maze_array.copy()
        num = -1
        # assert False
        # print(self.sim.model.body_pos)
        # self.sim.data.geom_xpos[27] = [1.55, 0.48, 0.42]
        for i in range(len(self.sim.model.body_pos)):
            self.sim.model.body_pos[i] = self.body_pos_initial_backup[i]
        # print(self.sim.model.body_pos) 
        # self.sim.model.body_pos[35] = [1.55, 0.48, 0.42]
        for col in range(10):
            for row in range(11):
                num += 1
                if maze_array[col][row] == 1:
                    # name = "object"+str(num)+":joint"
                    # object_qpos = self.sim.data.get_joint_qpos(name)
                    # object_qpos[0] = 1.08
                    # object_qpos[1] = 0.5
                    # object_qpos[0] += col*0.049
                    # object_qpos[1] += row*0.05
                    # object_qpos[2] = 0.42
                    self.sim.model.body_pos[num+35] = [1.08 + col*0.049, 0.5 + row*0.05, 0.42]

    # def randomize_environ_room_env(self):
    #     index = self.index
    #     # entry = (solid1wall_y,solid2wall_y,hollow1wall_y,hollow2wall_y,hollow3wall_y,hollow4wall_y,door1_x,door_1_y,door2_x,door_2_y)
    #     wallList = [(-2.85,-2.65,-2.75,-2.75,-2.715,-2.01,-2.75,-2.75,-2.75,-2.75),(0.85,0.65,-2.75,-2.75,-2.715,-2.01,-2.75,-2.75,-2.75,-2.75),
    #     (0.65, 0.85,-2.75,-2.75,-2.715,-2.01,-2.75,-2.75,-2.75,-2.75),(-2.85,-2.65,0.75,0.75,0.715,1.01,-2.75,-2.75,-2.75,-2.75),
    #     (-2.85,-2.65,0.75,0.75,-2.715,-2.01,-2.75,-2.75,-2.75,-2.75),(-2.85,-2.65,-2.75,-2.75,0.715,1.01,1.315,0.75,-2.75,-2.75),
    #     (-2.85,-2.65,0.75,-2.75,-2.715,1.01,-2.75,-2.75,-2.75,-2.75),(-2.85,-2.65,-2.75,0.75,-2.715,1.01,-2.75,-2.75,-2.75,-2.75),
    #     (-2.85,-2.65,-2.75,0.75,0.715,-2.01,-2.75,-2.75,-2.75,-2.75),(-2.85,-2.65,0.75,-2.75,0.715,-2.01,-2.75,-2.75,-2.75,-2.75),
    #     (-2.85,-2.65,0.75,0.75,-2.715,1.01,-2.75,-2.75,-2.75,-2.75),(-2.85,-2.65,0.75,0.75,0.715,-2.01,-2.75,-2.75,-2.75,-2.75),
    #     (-2.85,-2.65,-2.75,0.75,0.715,1.01,-2.75,-2.75,-2.75,-2.75),(-2.85,-2.65,0.75,-2.75,0.715,1.01,-2.21,-2.75,-2.75,-2.75),
    #     (-2.85,-2.65,0.75,0.75,-2.715,-2.01,1.18,0.75,-2.3,-2.63),(-2.85,-2.65,0.75,0.75,-2.715,-2.01,1.45,0.75,-2.3,-2.63),
    #     (-2.85,-2.65,0.75,0.75,-2.715,1.01,1.18,0.75,-2.75,-2.75),(-2.85,-2.65,0.75,-2.75,0.715,1.01,-2.75,-2.75,1.3,0.63),
    #     (-2.85,-2.65,0.75,0.75,0.715,-2.01,1.18,0.75,-2.75,-2.75),(-2.85,-2.65,-2.75,0.75,0.715,1.01,-2.75,-2.75,1.3,0.63),]   

    #     # index = 4
    #     self.polygons = []
    #     count = -1
    #     for i in self.obstacle_list[index]:
    #         count += 1
    #         if i == 1:
    #             self.polygons.append(self.polygons_list[count])

    #     object_qpos_s1 = self.sim.data.get_joint_qpos('solid1wall')
    #     object_qpos_s2 = self.sim.data.get_joint_qpos('solid2wall')
    #     object_qpos_h1 = self.sim.data.get_joint_qpos('hollow1wall')
    #     object_qpos_h2 = self.sim.data.get_joint_qpos('hollow2wall')
    #     object_qpos_h3 = self.sim.data.get_joint_qpos('hollow3wall')
    #     object_qpos_h4 = self.sim.data.get_joint_qpos('hollow4wall')
    #     object_qpos_d1 = self.sim.data.get_joint_qpos('door1')
    #     object_qpos_d2 = self.sim.data.get_joint_qpos('door2')

    #     object_qpos_s1[1] = wallList[index][0]
    #     object_qpos_s2[1] = wallList[index][1]
    #     object_qpos_h1[1] = wallList[index][2]
    #     object_qpos_h2[1] = wallList[index][3]
    #     object_qpos_h3[1] = wallList[index][4]
    #     object_qpos_h4[1] = wallList[index][5]
    #     object_qpos_d1[0] = wallList[index][6]
    #     object_qpos_d1[1] = wallList[index][7]
    #     object_qpos_d2[0] = wallList[index][8]
    #     object_qpos_d2[1] = wallList[index][9]
    #     if index != 1 and index != 2:
    #         object_qpos_d1[2] = 0.4
    #         object_qpos_d2[2] = 0.4

    #     self.sim.data.set_joint_qpos('solid1wall', object_qpos_s1)
    #     self.sim.data.set_joint_qpos('solid2wall', object_qpos_s2)
    #     self.sim.data.set_joint_qpos('hollow1wall', object_qpos_h1)
    #     self.sim.data.set_joint_qpos('hollow2wall', object_qpos_h2)
    #     self.sim.data.set_joint_qpos('hollow3wall', object_qpos_h3)
    #     self.sim.data.set_joint_qpos('hollow4wall', object_qpos_h4)
    #     self.sim.data.set_joint_qpos('door1', object_qpos_d1)
    #     self.sim.data.set_joint_qpos('door2', object_qpos_d2)

    def if_collision(self):
        for i in range(self.sim.data.ncon):
            flag_contact = 0
            contact = self.sim.data.contact[i]
            name1 = self.sim.model.geom_id2name(contact.geom1)
            name2 = self.sim.model.geom_id2name(contact.geom2)
            if name1 is None or name2 is None:
                break
            for num in range(3,6):
                if ("robot0:l_gripper_finger_link" == name1 and "G"+str(num) in name2) or ("robot0:r_gripper_finger_link" == name2 and "G"+str(num) in name1):
                    # print(name1, name2)
                    return 1
            # if name1 is None or name2 is None:
            #     break
            # if (("robot0:l_gripper_finger_link" == name1 and "object" in name2) or ("robot0:l_gripper_finger_link" == name2 and "object" in name2)) or(("robot0:r_gripper_finger_link" == name1 and "object" in name2) or ("robot0:r_gripper_finger_link" == name2 and "object" in name2)):
            #     # print('contact', i)
            #     # print('geom1', name1[:6])
            #     # print('geom2', name2[:6])
            #     return True
        return 0

    def collision_with_initial_rope(self):
        old_value = self.collision_with_initial_rope
        return old_value

    def num_walls_collision(self):
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            name1 = self.sim.model.geom_id2name(contact.geom1)
            name2 = self.sim.model.geom_id2name(contact.geom2)
            if name1 is None or name2 is None:
                break
            if "robot0:l_gripper_finger_link" == name1 and "object" in name2:
                if name2 not in self.curr_eps_num_wall_collisions:
                    self.curr_eps_num_wall_collisions.append(name2)
            elif "robot0:l_gripper_finger_link" == name2 and "object" in name1:
                if name1 not in self.curr_eps_num_wall_collisions:
                    self.curr_eps_num_wall_collisions.append(name1)
            elif "robot0:r_gripper_finger_link" == name1 and "object" in name2:
                if name2 not in self.curr_eps_num_wall_collisions:
                    self.curr_eps_num_wall_collisions.append(name2)
            elif "robot0:r_gripper_finger_link" == name2 and "object" in name1:
                if name1 not in self.curr_eps_num_wall_collisions:
                    self.curr_eps_num_wall_collisions.append(name1)

        return len(self.curr_eps_num_wall_collisions)

    def _sample_goal(self, index = -1):
        if self.has_object:
            goal = self.get_goal_pos()
            # while self.check_overlap(goal):
            #     goal = self.get_goal_pos()
            #(1.06,1.23)
            #(0.57,0.92)
            # Polygon([(1.06, 0.73),(1.12,0.73),(1.12,0.77),(1.06, 0.77)]),#verti_up   
            # Polygon([(1.215, 0.73),(1.38,0.73),(1.38,0.77),(1.215, 0.77)]),#verti_mid  
            # Polygon([(1.48, 0.73),(1.54,0.73),(1.54,0.77),(1.48, 0.77)]),#verti_down 
            # Polygon([(1.28, 0.668),(1.32,0.668),(1.32, 0.835),(1.28, 0.835)])#hori_mid   
            # goal[:3] = [1.35, 0.7, 0.42]
            # goal[:3] = [1.1, ., 0.42]
        else:
            # goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal = self.get_goal_pos(index)
            # while self.check_overlap(goal):
            #     goal = self.get_goal_pos()
            # goal[:3] = [1.55, 0.48, 0.42]
            # goal[2] = 0.42
            '''
            goal_ranges:
            x:[1.05, 1.55], diff=(0.25), offset=(1.3)
            y:[0.48, 1.02], diff=(0.27), offset=(0.75)
            z:[0.41, 0.7],  diff=(0.145), offset=(0.555)

            self.max_u = [0.24, 0.27, 0]
            self.action_offset = [1.29, 0.81, 0.43]
            u = self.action_offset + (self.max_u * u )
            
            '''
        self.goal = goal.copy()
        # print(self.goal)            
        # self.set_subgoal('subgoal_4', [1.32, 0.835, 0.45])
        return goal.copy()

    def get_possible_goals(self):
        max_u = [0.25, 0.27, 0.145]
        action_offset = [1.3, 0.75, 0.555]
        outer_array = np.arange(action_offset[0]-max_u[0], action_offset[0]+max_u[0]+(2*max_u[0])/(10), (2*max_u[0])/(9))
        inner_array = np.arange(action_offset[1]-max_u[1], action_offset[1]+max_u[1]+(2*max_u[1])/(11), (2*max_u[1])/(10))
        return outer_array, inner_array

    def get_goal_pos(self, index = -1):
        if index != -1:
            random.seed(index)

        goal_index = index#random.randrange(self.obs.shape[0])
        curr_obs_list = self.obs[goal_index]
        # curr_acs_list = self.acs[goal_index]
        goal_obs = curr_obs_list[len(curr_obs_list)-1]['desired_goal'].copy()
        # print(len(curr_obs_list))
        # assert False
        for num in range(5,20):
            self.set_subgoal('subgoal_goal_'+str(num), np.array([goal_obs[2*(num-5)], goal_obs[2*(num-5)+1], 0.42]))
        # for num in range(5,20):
        #     self.set_subgoal('subgoal_goal_'+str(num), goal_obs[3*(num-5):3*(num-4)])
        if index != -1:
            random.seed(self.rank_seed)
        # goal_obs = np.zeros((30))
        return goal_obs.copy()

    def get_maze_array(self):
        return self.maze_array.copy()

    def get_maze_array_simple(self):
        maze = self.maze_array.copy()
        maze[:,-1] = 0
        maze[:,0] = 0
        maze[-1,:] = 0
        maze[0,:] = 0
        maze[-2,:] = 0
        return maze

    def check_overlap(self, point):
        xx = Point(point[:2])
        for polygon in self.polygons:
            if xx.within(polygon):
                return 1
        return 0

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def get_current_gripper_pos(self):
        return self.sim.data.get_site_xpos('robot0:grip').copy()

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_target = np.array([1.13, 0.65, 0.44])
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        self.initial_gripper_xpos = np.array([1.3, 0.75, 0.44])
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def capture(self,depth=False):
        if self.viewer == None:
            pass
        else:
            self.viewer.cam.fixedcamid = 3
            self.viewer.cam.type = const.CAMERA_FIXED
            width, height = 1920, 1080
            img = self._get_viewer().read_pixels(width, height, depth=depth)
            # if not depth:
                # img = img[350:623,726:1193]
                # img = cv2.resize(img, (50,50))
            # print(img[:].shape)
            # # depth_image = img[:][:][1][::-1] # To visualize the depth image(depth=True)
            # # rgb_image = img[:][:][0][::-1] # To visualize the depth image(depth=True)
            # depth_image = img[:][:][1][::-1]
            # rgb_image = img[:][:][0][::-1]
            if depth:
                rgb_image = img[0][::-1]
                rgb_image = rgb_image[458:730,726:1193]
                rgb_image = cv2.resize(rgb_image, (50,50))
                depth_image = np.expand_dims(img[1][::-1],axis=2)
                depth_image = depth_image[458:730,726:1193]
                depth_image = cv2.resize(depth_image, (50,50))
                depth_image = np.reshape(depth_image, (50,50,1))
                rgbd_image = np.concatenate((rgb_image,depth_image),axis=2)
                return rgbd_image
            else:
                return img[::-1]

    def is_achievable(self, curr_pos, pred_pos):
        dist = np.linalg.norm(curr_pos - pred_pos)
        threshold = 0.35
        if dist < threshold:
            return 0
        else:
            return -1








