from collections import deque
import tensorflow as tf
import numpy as np
import pickle
import random
from mujoco_py import MujocoException
from tqdm import tqdm
import gym
import sys
import pickle
from util import convert_episode_to_batch_major, store_args
try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

Qs = []
def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class RolloutWorkerOriginal:

    @store_args
    def __init__(self, make_env, policyList, dims, logger, hrl_imitation_loss, num_upper_demos, adversarial_loss, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0, raps=0,
                 random_eps=0, history_len=100, render=False, sac=False, predictor_loss=0, populate=0, reward_model=0, lower_reward_model=0, 
                 is_multiple_env=False, hac=0, hier=0, dac=0, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        self.is_multiple_env = is_multiple_env
        self.envs = [make_env() for _ in range(rollout_batch_size)]
        self.env_populate = make_env()
        assert self.T > 0
        self.compute_Q = False
        self.sac = sac
        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        # self.success_history_non_contact_walls = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)
        self.r_upper_history = deque(maxlen=history_len)
        self.r_lower_history = deque(maxlen=history_len)
        self.num_upper_demos = num_upper_demos

        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.reset_all_rollouts()
        self.testingNow = kwargs['testingNow']
        self.clear_history()
        self.hrl_imitation_loss = hrl_imitation_loss
        self.adversarial_loss = adversarial_loss
        self.global_curr_eps_step_count = 0
        self.num_collision_with_wall = 0
        self.num_wall_collision = 0
        self.curr_eps_step_count = 0
        self.first_success_flag = 0
        self.hac = hac
        self.dac = dac
        self.hier = hier
        self.populate_once_done = 0
        self.raps = raps
        if self.hac:
            self.hac_iterate = 3
        self.obs_final, self.achieved_goals_final, self.acts_final, self.goals_final, self.successes_final, self.rewards_final, self.env_indexes_final = [], [], [], [], [], [], []
        self.obs_complete, self.achieved_goals_complete, self.acts_complete, self.goals_complete, self.successes_complete, self.rewards_complete, self.env_indexes_complete = [], [], [], [], [], [], []
        self.max_u = [0.25, 0.27, 0.145]
        self.action_offset = [1.3, 0.75, 0.555]
        self.global_episode = dict(u=[], success=[])
        self.is_successful_values = []
        self.dist_values = []
        self.populate = populate
        self.reward_model = reward_model
        self.lower_reward_model = lower_reward_model
        self.predictor_loss = predictor_loss
        self.demoData_obs_list_train = []#demoData_1['obs'][:5000].copy()
        self.demoData_obs_temp_train = []
        self.current_timestep = -1
        for num in range(len(self.demoData_obs_list_train)):
            current_obs = self.demoData_obs_list_train[num][-1].copy()
            current_obs['desired_goal'] = current_obs['observation'].copy()
            self.demoData_obs_temp_train.append(np.array([current_obs]))

        if self.populate:
            if 'Rope' in str(self.envs[0]):
                demoDataFile = './data/robotic_rope_dataset.npz'
            elif 'Bin' in str(self.envs[0]):
                demoDataFile = './data/bin_dataset.npz'
            elif 'Hollow' in str(self.envs[0]):
                demoDataFile = './data/hollow_dataset.npz'
            elif 'Pick' in str(self.envs[0]):
                demoDataFile = './data/pick_dataset.npz'
            elif 'kitchen' in str(self.envs[0]):
                demoDataFile = './data/kitchen_dataset.npz'
            else:
                demoDataFile = './data/maze_dataset.npz'
            demoData = np.load(demoDataFile, allow_pickle=True)
            self.demoData_obs_full = demoData['obs'].copy()
            self.demoData_acs_full = demoData['acs'].copy()
            if 'Rope' in str(self.envs[0]):
                self.demoData_states_full = demoData['states'].copy()
            del demoData

    def reset_rollout(self, i, seed=0, is_train=False):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        obs = self.envs[i].reset()
        if 'kitchen' not in str(self.envs[0]):
            obs = self.envs[i].env.setIndex(seed)
        self.initial_o[i] = obs['observation']
        self.initial_ag[i] = obs['achieved_goal']
        self.g[i] = obs['desired_goal']

    def reset_all_rollouts(self, seed=0, is_train=False):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i, seed, is_train)

    def generate_rollouts(self, seed=0, epoch_num=0, is_train=False):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts(seed, is_train)
        self.global_curr_eps_step_count = 0
        self.curr_eps_step_count = 0
        self.first_success_flag = 0
        self.num_collision_with_wall = 0
        self.num_wall_collision = 0
        self.subgoals_count = 0
        self.subgoals_array = []
        # compute Observations
        policyList = self.policyList
        num_hrl_layers = len(policyList)
        self.num_hrl_layers = num_hrl_layers
        self.goal_array = [None for i in range(num_hrl_layers)]
        self.hrl_initial_obs = [None for i in range(num_hrl_layers)]
        self.hrl_initial_ag = [None for i in range(num_hrl_layers)]
        for i in range(len(policyList)):
            self.hrl_initial_obs[i] = self.initial_o.copy()
            self.hrl_initial_ag[i] = self.initial_ag.copy()
        self.goal_array[num_hrl_layers - 1] = self.g.copy()
        self.policyList = policyList

        if 'Rope' in str(self.envs[0]):
            # predict_loss_num = 5
            populate_num = 150
        elif 'Pick' in str(self.envs[0]):
            # predict_loss_num = 5
            populate_num = 50
        elif 'Maze' in str(self.envs[0]):
            # predict_loss_num = 10
            populate_num = 50
        elif 'kitchen' in str(self.envs[0]):
            populate_num = 100

        if self.populate and is_train and epoch_num % populate_num == 0 and self.num_hrl_layers > 1:
            '''
            Calls populate function which polulates the upper level demo buffer after p timesteps
            '''
            if 'Rope' in str(self.envs[0]):
                self.populate_upper_demo_buffer_rope_uns(self.policyList[0])
            elif 'Pick' in str(self.envs[0]):
                self.populate_upper_demo_buffer_pick_uns(self.policyList[0])
            elif 'Maze' in str(self.envs[0]):
                self.populate_upper_demo_buffer_maze_uns(self.policyList[0])
            elif 'kitchen' in str(self.envs[0]):
                self.populate_upper_demo_buffer_kitchen_uns(self.policyList[0])
        self.rec_rollout_generator(num_hrl_layers - 1, seed, epoch_num)
        return self.is_successful_values, self.dist_values
        

    def compute_reward_dense(self, achieved_goal, goal):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        return -d

    def compute_direction_reward(self, obs, action, subgoal):
        # Reward if action taken is in direction of subgoal
        sign_action = np.sign(action[:2])
        temp = (subgoal - obs[:3])[:2]
        sign_obs = np.sign(temp)
        differ = sign_obs - sign_action
        reward = 0
        if differ[0] != 0:
            reward -= 5
        if differ[1] != 0:
            reward -= 5
        return reward

    def goal_distance(goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def rope_action_to_obs_encoder(self, u_temp):
        ''' Encodes rope actions to observations for higher level policy subgoal prediction
        '''
        max_angle = 45
        x_min = 1.05
        x_max = 1.55
        y_min = 0.48
        y_max = 1.02
        rope_dist = 0.0299
        initial_pos = [1.11025568, 0.75]
        u = np.zeros(30)
        u[0] = initial_pos[0]
        u[1] = initial_pos[1]
        u[2] = 1.14
        u[3] = 0.75
        initial_angle = 0
        prev_angle = initial_angle
        for num in range(2,15):
            drag_theta = prev_angle + max_angle*(u_temp[num])
            prev_angle = drag_theta
            u[2*num] = u[2*(num-1)] + rope_dist*np.cos(np.radians(drag_theta))*1.0
            u[2*num+1] = u[2*(num-1)+1] + rope_dist*np.sin(np.radians(drag_theta))*1.0
            u[2*num] = max(x_min, u[2*num])
            u[2*num] = min(x_max, u[2*num])
            u[2*num+1] = max(y_min, u[2*num+1])
            u[2*num+1] = min(y_max, u[2*num+1])
        return u.copy()
    
    def populate_upper_demo_buffer_pick_uns(self, policy):
        ''' Populates upper level demo buffer in pick and place environment
         '''
        num_episodes = self.num_upper_demos
        observations = []#[[] for i in range(num_episodes)]
        actions = []#[[] for i in range(num_episodes)]
        policy.clear_buffer_upper()
        self.env_populate._max_episode_steps = 2000
        max_u = [0.24, 0.27, 0.145]
        action_offset = [1.29, 0.75, 0.555]
        self.env_populate.reset()
        now_selected_subgoal = 28
        now_current_position = 20
        for eps_index in range(num_episodes):
            transition = 0
            prev_transition = 0
            episodeAcs = []
            episodeObs = []
            self.env_populate.env.setIndex(eps_index)
            if len(self.demoData_obs_full[eps_index]) == 0:
                # print("Not reached")
                continue
            demoData_obs = self.demoData_obs_full[eps_index]
            demoData_acs = self.demoData_acs_full[eps_index]
            demoData_obs_epsd = demoData_obs
            demoData_acs_epsd = demoData_acs
            now_selected_subgoal = 28
            now_current_position = 20

            episode_len = len(demoData_obs_epsd)-1

            dist_thresh = 0.013
            prev_transition = transition
            while transition < episode_len-1:
                Q_values = np.zeros(episode_len)
                next_transition = transition
                now_current_position +=1 
                demoData_obs_epsd_transition = demoData_obs_epsd[transition]
                obs = demoData_obs_epsd_transition.get('observation')
                achieved_goal = demoData_obs_epsd_transition.get('achieved_goal')
                obs_list, achieved_goal_list, current_goal_list = [], [], []
                old_transition = next_transition
                while next_transition < episode_len-1:
                    next_transition += 1
                    current_goal = demoData_obs_epsd[next_transition].get('observation')[:3]
                    obs_list.append(obs)
                    achieved_goal_list.append(achieved_goal)
                    current_goal_list.append(current_goal)
                ret = policy.get_Q_sac(obs_list, achieved_goal_list, current_goal_list)
                array_index = 0
                for i in range(old_transition+1, next_transition+1):
                    Q_values[i] = list(ret[0][array_index])[0]
                    array_index += 1

                min_val = min(Q_values)
                if min_val < 0:
                    Q_values -= min_val
                max_val = max(Q_values)
                Q_values /= max_val
                Q_values = (Q_values*100).astype(int)
                dist_threshold = 0
                Q_values[Q_values < dist_threshold] = 0
                Q_values[Q_values >= dist_threshold] = 1
                if transition:
                    Q_values[:transition+1] = 0

                final_transition = episode_len-1
                transition_flag = -1
                for final_transition in range(transition+1, len(Q_values)):
                    transition_flag += 1
                    if transition_flag >= self.T or Q_values[final_transition] == 0:
                        now_selected_subgoal += 1
                        break

                goal_lower = demoData_obs_epsd[final_transition].get('achieved_goal')
                action_temp = (goal_lower - action_offset) / [x for x in max_u]
                action = np.zeros(4)
                action[:3] = action_temp
                for current_transition in range(prev_transition, final_transition):
                    for next_transition in range(current_transition+1, final_transition):
                        episodeObs = []
                        episodeAcs = []
                        demoData_obs_epsd[current_transition]['achieved_goal'] = goal_lower.copy()
                        episodeObs.append(demoData_obs_epsd[current_transition])
                        episodeAcs.append(np.array([action]))
                        demoData_obs_epsd[next_transition]['achieved_goal'] = goal_lower.copy()
                        episodeObs.append(demoData_obs_epsd[next_transition])
                        actions.append(episodeAcs)
                        observations.append(np.array([episodeObs]))
                transition = final_transition
                prev_transition = transition

        if not self.populate_once_done:
            self.populate_once_done = 1
            update_stats = True
        else:
            update_stats = False
        policy.initDemoBufferUpperMultipleCurriculum(observations, actions, update_stats)
        self.env_populate.close()

    def populate_upper_demo_buffer_maze_uns(self, policy):
        ''' Populates upper level demo buffer in maze navigation environment
         '''
        num_episodes = self.num_upper_demos
        observations = []#[[] for i in range(num_episodes)]
        actions = []#[[] for i in range(num_episodes)]
        policy.clear_buffer_upper()
        self.env_populate._max_episode_steps = 2000
        max_u = [0.24, 0.27, 0.145]
        action_offset = [1.29, 0.75, 0.555]
        self.env_populate.reset()
        now_selected_subgoal = 28
        now_current_position = 20
        for eps_index in range(num_episodes):
            transition = 0
            prev_transition = 0
            episodeAcs = []
            episodeObs = []
            self.env_populate.env.setIndex(eps_index)
            maze_array = self.env_populate.env.get_maze_array().ravel().tolist().copy()
            if len(self.demoData_obs_full[eps_index]) == 0:
                # print("Not reached")
                continue
            demoData_obs = self.demoData_obs_full[eps_index][0]
            demoData_acs = self.demoData_acs_full[eps_index][0]
            demoData_obs_epsd = demoData_obs
            demoData_acs_epsd = demoData_acs
            now_selected_subgoal = 28
            now_current_position = 20

            episode_len = len(demoData_obs_epsd)-1

            dist_thresh = 0.013
            prev_transition = transition
            while transition < episode_len-1:
                Q_values = np.zeros(episode_len)
                next_transition = transition
                now_current_position +=1 
                demoData_obs_epsd_transition = demoData_obs_epsd[transition]
                obs = demoData_obs_epsd_transition.get('observation')
                obs = np.array(obs[:3].tolist() + maze_array.copy()).copy()
                achieved_goal = demoData_obs_epsd_transition.get('achieved_goal')
                obs_list, achieved_goal_list, current_goal_list = [], [], []
                old_transition = next_transition
                while next_transition < episode_len-1:
                    next_transition += 1
                    current_goal = demoData_obs_epsd[next_transition].get('observation')[:3]
                    obs_list.append(obs)
                    achieved_goal_list.append(achieved_goal)
                    current_goal_list.append(current_goal)
                ret = policy.get_Q_sac(obs_list, achieved_goal_list, current_goal_list)
                array_index = 0
                for i in range(old_transition+1, next_transition+1):
                    Q_values[i] = list(ret[0][array_index])[0]
                    array_index += 1

                min_val = min(Q_values)
                if min_val < 0:
                    Q_values -= min_val
                max_val = max(Q_values)
                Q_values /= max_val
                Q_values = (Q_values*100).astype(int)
                if 'Maze' in str(self.envs[0]):
                    dist_threshold = 10
                else:
                    dist_threshold = 0
                Q_values[Q_values < dist_threshold] = 0
                Q_values[Q_values >= dist_threshold] = 1
                if transition:
                    Q_values[:transition+1] = 0

                final_transition = episode_len-1
                transition_flag = -1
                for final_transition in range(transition+1, len(Q_values)):
                    transition_flag += 1
                    if transition_flag >= self.T or Q_values[final_transition] == 0:
                        now_selected_subgoal += 1
                        break

                goal_lower = demoData_obs_epsd[final_transition].get('achieved_goal')
                action_temp = (goal_lower - action_offset) / [x for x in max_u]
                action = np.zeros(4)
                action[:3] = action_temp
                for current_transition in range(prev_transition, final_transition):
                    for next_transition in range(current_transition+1, final_transition):
                        episodeObs = []
                        episodeAcs = []
                        demoData_obs_epsd[current_transition]['achieved_goal'] = goal_lower.copy()
                        episodeObs.append(demoData_obs_epsd[current_transition])
                        episodeAcs.append(np.array([action]))
                        demoData_obs_epsd[next_transition]['achieved_goal'] = goal_lower.copy()
                        episodeObs.append(demoData_obs_epsd[next_transition])
                        actions.append(episodeAcs)
                        observations.append(np.array([episodeObs]))
                transition = final_transition
                prev_transition = transition

        if not self.populate_once_done:
            self.populate_once_done = 1
            update_stats = True
        else:
            update_stats = False
        policy.initDemoBufferUpperMultipleCurriculum(observations, actions, update_stats)
        self.env_populate.close()

    def populate_upper_demo_buffer_kitchen_uns(self, policy):
        ''' Populates upper level demo buffer in kitchen environment
         '''
        num_episodes =len(self.demoData_acs_full)
        observations = []#[[] for i in range(num_episodes)]
        actions = []#[[] for i in range(num_episodes)]
        policy.clear_buffer_upper()
        self.env_populate._max_episode_steps = 4000
        max_u = [1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,0.88,0.01,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,0.75,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6]
        action_offset = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.env_populate.reset()
        now_selected_subgoal = 28
        now_current_position = 20
        for eps_index in range(num_episodes):
            transition = 0
            prev_transition = 0
            episodeAcs = []
            episodeObs = []
            demoData_obs = self.demoData_obs_full[eps_index]
            now_selected_subgoal = 28
            now_current_position = 20
            demoData_obs_epsd = demoData_obs
            episode_len = len(demoData_obs_epsd)-1
            prev_transition = transition
            while transition < episode_len-1:
                Q_values = np.zeros(episode_len)
                next_transition = transition
                now_current_position +=1 
                demoData_obs_epsd_transition = demoData_obs_epsd[transition]
                obs = demoData_obs_epsd_transition.get('observation')
                achieved_goal = demoData_obs_epsd_transition.get('achieved_goal')
                obs_list, achieved_goal_list, current_goal_list = [], [], []
                old_transition = next_transition
                while next_transition < episode_len-1:
                    next_transition += 1
                    current_goal = demoData_obs_epsd[next_transition].get('achieved_goal')
                    obs_list.append(obs)
                    achieved_goal_list.append(achieved_goal)
                    current_goal_list.append(current_goal)
                ret = policy.get_Q_sac(obs_list, achieved_goal_list, current_goal_list)
                array_index = 0
                for i in range(old_transition+1, next_transition+1):
                    Q_values[i] = list(ret[0][array_index])[0]
                    array_index += 1

                min_val = min(Q_values)
                if min_val < 0:
                    Q_values -= min_val
                max_val = max(Q_values)
                Q_values /= max_val
                Q_values = (Q_values*100).astype(int)
                dist_threshold = 0
                Q_values[Q_values < dist_threshold] = 0
                Q_values[Q_values >= dist_threshold] = 1
                if transition:
                    Q_values[:transition+1] = 0

                final_transition = episode_len-1
                transition_flag = -1
                for final_transition in range(transition+1, len(Q_values)):
                    transition_flag += 1
                    if Q_values[final_transition] == 0:
                        now_selected_subgoal += 1
                        break

                goal_lower_1 = demoData_obs_epsd[final_transition].get('achieved_goal').copy()
                goal_lower_1 = (goal_lower_1 - action_offset) / [x for x in max_u]

                obs_prediction = goal_lower_1.copy()
                achieved_goal_reference = np.round(obs_prediction.copy(), decimals = 2)
                goal_lower = np.zeros_like(achieved_goal_reference).astype(np.float32)
                for element in self.envs[0].env.TASK_ELEMENTS:
                    element_idx = self.envs[0].env.OBS_ELEMENT_INDICES[element]
                    first_flag = 0
                    for final_index in element_idx:
                        if first_flag:
                            goal_lower[final_index] = -np.array(max_u)[final_index]
                            break
                        if np.sign(achieved_goal_reference[final_index]) == -np.sign(np.array(max_u)[final_index]):
                            first_flag = 1
                            goal_lower[final_index] = -np.array(max_u)[final_index]


                action_temp = goal_lower
                action = action_temp.copy()
                for current_transition in range(prev_transition, final_transition):
                    if current_transition+1 == final_transition and current_transition+1 < episode_len:
                        next_transition = current_transition+1
                        episodeObs = []
                        episodeAcs = []
                        episodeObs.append(demoData_obs_epsd[current_transition])
                        episodeAcs.append(np.array([action]))
                        episodeObs.append(demoData_obs_epsd[next_transition])
                        actions.append(episodeAcs)
                        observations.append(np.array([episodeObs]))
                    else:
                        for next_transition in range(current_transition+1, final_transition):
                            episodeObs = []
                            episodeAcs = []
                            episodeObs.append(demoData_obs_epsd[current_transition])
                            episodeAcs.append(np.array([action]))
                            episodeObs.append(demoData_obs_epsd[next_transition])
                            actions.append(episodeAcs)
                            observations.append(np.array([episodeObs]))
                transition = final_transition
                prev_transition = transition

        if not self.populate_once_done:
            self.populate_once_done = 1
            update_stats = True
        else:
            update_stats = False
        policy.initDemoBufferUpperMultipleCurriculum(observations, actions, update_stats)
        self.env_populate.close()

    def populate_upper_demo_buffer_rope_uns(self, policy):
        ''' Populates upper level demo buffer in rope manipulation environment
         '''
        num_episodes = self.num_upper_demos
        observations = []#[[] for i in range(num_episodes)]
        actions = []#[[] for i in range(num_episodes)]
        states = []#[[] for i in range(num_episodes)]
        policy.clear_buffer_upper()
        self.env_populate._max_episode_steps = 4000
        max_u = np.array([])
        action_offset = np.array([])
        for i in range(15):
            max_u = np.append(max_u, [0.25, 0.27], axis=0)
        for i in range(15):
            action_offset = np.append(action_offset, [1.3, 0.75], axis=0)
        self.env_populate.reset()
        now_selected_subgoal = 28
        now_current_position = 20
        for eps_index in range(num_episodes):
            self.env_populate.env.setIndex(eps_index)
            transition = 0
            prev_transition = 0
            episodeAcs = []
            episodeObs = []
            episodeStates = []
            if len(self.demoData_obs_full[eps_index]) == 0 or len(self.demoData_obs_full[eps_index])==50:
                # print("Not reached", len(demoData_obs_full[index]))
                continue
            demoData_obs = self.demoData_obs_full[eps_index]
            demoData_states = self.demoData_states_full[eps_index]
            now_selected_subgoal = 28
            now_current_position = 20
            demoData_obs_epsd = demoData_obs
            demoData_obs_states = demoData_states
            episode_len = len(demoData_obs_epsd)-1
            prev_transition = transition
            while transition < episode_len-1:
                Q_values = np.zeros(episode_len)
                next_transition = transition
                now_current_position +=1 
                demoData_obs_epsd_transition = demoData_obs_epsd[transition]
                obs = demoData_obs_epsd_transition.get('observation')
                achieved_goal = demoData_obs_epsd_transition.get('achieved_goal')
                obs_list, achieved_goal_list, current_goal_list = [], [], []
                old_transition = next_transition
                while next_transition < episode_len-1:
                    next_transition += 1
                    current_goal = demoData_obs_epsd[next_transition].get('achieved_goal')
                    obs_list.append(obs)
                    achieved_goal_list.append(achieved_goal)
                    current_goal_list.append(current_goal)
                ret = policy.get_Q_sac(obs_list, achieved_goal_list, current_goal_list)
                array_index = 0
                for i in range(old_transition+1, next_transition+1):
                    Q_values[i] = list(ret[0][array_index])[0]
                    array_index += 1

                min_val = min(Q_values)
                if min_val < 0:
                    Q_values -= min_val
                max_val = max(Q_values)
                Q_values /= max_val
                Q_values = (Q_values*100).astype(int)
                dist_threshold = 0
                Q_values[Q_values < dist_threshold] = 0
                Q_values[Q_values >= dist_threshold] = 1
                if transition:
                    Q_values[:transition+1] = 0

                final_transition = episode_len-1
                transition_flag = -1
                for final_transition in range(transition+1, len(Q_values)):
                    transition_flag += 1
                    if transition_flag >= self.T or Q_values[final_transition] == 0:

                        now_selected_subgoal += 1
                        break
                goal_lower = demoData_obs_epsd[final_transition].get('achieved_goal')
                action_temp = (goal_lower - action_offset) / [x for x in max_u]
                action = action_temp.copy()
                for current_transition in range(prev_transition, final_transition):
                    for next_transition in range(current_transition+1, final_transition):
                        episodeObs = []
                        episodeAcs = []
                        demoData_obs_epsd[current_transition]['achieved_goal'] = goal_lower.copy()
                        episodeObs.append(demoData_obs_epsd[current_transition])
                        episodeAcs.append(np.array([action]))
                        demoData_obs_epsd[next_transition]['achieved_goal'] = goal_lower.copy()
                        episodeObs.append(demoData_obs_epsd[next_transition])
                        actions.append(episodeAcs)
                        observations.append(np.array([episodeObs]))
                transition = final_transition
                prev_transition = transition
        if not self.populate_once_done:
            self.populate_once_done = 1
            update_stats = True
        else:
            update_stats = False
        policy.initDemoBufferUpperMultipleCurriculum(observations, actions, update_stats)
        # print('Time taken', time.time() - start)
        self.env_populate.close()
    
    def rec_rollout_generator(self, hrl_layer_num, seed=0, epoch_num = -1):
        '''  Generates rollouts 
         '''
        global Qs
        num_hrl_layers = len(self.policyList)
        env_index = seed
        if 'kitchen' in str(self.envs[0]):
            max_u = [1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,0.88,0.01,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,0.75,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6]
            action_offset = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        else:
            max_u = [0.25, 0.27, 0.145]
            action_offset = [1.3, 0.75, 0.555]
        if num_hrl_layers > 1:
            episode_len = self.T * self.T
        else:
            episode_len = self.T

        if hrl_layer_num > 0:
            # Higher level policy
            # Observations
            hrl_obs = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
            hrl_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
            curr_obs = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # achieved goals
            hrl_obs_2 = np.empty((self.rollout_batch_size, self.dims['o']))
            hrl_ag_2 = np.empty((self.rollout_batch_size, self.dims['g']))

            hrl_obs = self.hrl_initial_obs[hrl_layer_num].copy()
            hrl_ag = self.hrl_initial_ag[hrl_layer_num].copy()
            curr_ag = self.hrl_initial_ag[hrl_layer_num].copy()

            # generate episodes for buffer
            obs, achieved_goals, acts, goals, successes, rewards, env_indexes, rewards_book = [], [], [], [], [], [], [], []
            if self.hac:
                rewards_hac2 = []
                achieved_goals_hac1 = []

            # generate episodes for demoBuffer1/inverseRL
            obs_2, achieved_goals_2, acts_2, goals_2, successes_2, rewards_2, env_indexes_2 = [], [], [], [], [], [], []

            final_upper_success = 0
            old_final_upper_success = 0
            successes_non_contact_walls = []
            info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
            infoG = {}
            for t in range(self.T):
                self.curr_eps_step_count += 1
                if self.sac == 0:
                    policy_output = self.policyList[hrl_layer_num].get_actions(hrl_obs[0], hrl_ag, self.goal_array[hrl_layer_num], t, self.demoData_obs_list_train, env_index = env_index, 
                    compute_Q=self.compute_Q,noise_eps=self.noise_eps if not self.exploit else 0.,
                    random_eps=self.random_eps if not self.exploit else 0.,use_target_net=self.use_target_net)
                else:
                    policy_output = self.policyList[hrl_layer_num].get_actions_sac(hrl_obs[0], hrl_ag, self.goal_array[hrl_layer_num], env_index = env_index,
                    compute_Q=self.compute_Q,noise_eps=self.noise_eps if not self.exploit else 0.,
                    random_eps=self.random_eps if not self.exploit else 0.,use_target_net=self.use_target_net, deterministic=self.testingNow)


                if t == self.T-1:
                    self.dist_values.append(self.policyList[hrl_layer_num].get_current_min_dist())
                if self.compute_Q:
                    u, Q = policy_output
                    Qs.append(Q)
                else:
                    u = policy_output

                if u.ndim == 1:
                    # The non-batched case should still have a reasonable shape.
                    u = u.reshape(1, -1)
                unscaled_u = u.copy()
                if self.raps and ('Bin' in str(self.envs[0]) or 'Pick' in str(self.envs[0]) or 'Hollow' in str(self.envs[0])):
                    u_temp1 = np.array(u[0][:3]).reshape(1,-1)
                    u_temp1 = action_offset + (max_u * u_temp1 )
                    u_temp2 = u[0][3]
                    u_raps = np.array([[u_temp1[0][0], u_temp1[0][1], u_temp1[0][2], u_temp2]])

                    u = np.array(u[0][:3]).reshape(1,-1)
                    u = action_offset + (max_u * u )
                else:
                    # Rescaling actions to make them lie in our range
                    if 'Rope' in str(self.envs[0]):
                        u_temp = u[0].copy()
                        u = self.rope_action_to_obs_encoder(u_temp)
                        u = np.array(u).reshape(1,-1)
                    else:
                        if 'kitchen' in str(self.envs[0]):
                            u = np.array(u[0]).reshape(1,-1)
                        else:
                            u = np.array(u[0][:3]).reshape(1,-1)
                        u = action_offset + (max_u * u )

                if 'Reach' in str(self.envs[0]):# or 'Pick' in str(self.envs[0]):
                    u[0][2]=0.42

                # reward_prediction = self.policyList[hrl_layer_num].get_rewards_sac_no_ag(hrl_obs[0], self.goal_array[hrl_layer_num], unscaled_u, env_index = env_index,
                #     compute_Q=self.compute_Q,noise_eps=self.noise_eps if not self.exploit else 0.,
                #     random_eps=self.random_eps if not self.exploit else 0.,use_target_net=self.use_target_net, deterministic=self.testingNow)
                # print('reward_prediction', reward_prediction)
                
                success = np.zeros(self.rollout_batch_size)
                success_non_contact_walls = np.zeros(self.rollout_batch_size)
                reward = np.zeros((self.rollout_batch_size, 1), np.float32)
                if self.hac:
                    reward_hac2 = np.zeros((self.rollout_batch_size, 1), np.float32)
                reward_book = np.zeros((self.rollout_batch_size, 1), np.float32)
                # compute new states and observations
                for i in range(self.rollout_batch_size):
                    try:
                        if self.raps  and ('Bin' in str(self.envs[0]) or 'Pick' in str(self.envs[0]) or 'Hollow' in str(self.envs[0])):
                            self.goal_array[hrl_layer_num-1] = u_raps[i].reshape(1,-1)
                        else:
                            self.goal_array[hrl_layer_num-1] = u[i].reshape(1,-1)
                        if 'kitchen' in str(self.envs[0]):
                            obs_temp = u[i]

                            obs_prediction = obs_temp.copy()
                            achieved_goal_reference = np.round(obs_prediction.copy(), decimals = 2)
                            goal_lower = np.zeros_like(achieved_goal_reference).astype(np.float32)
                            for element in self.envs[i].env.TASK_ELEMENTS:
                                element_idx = self.envs[i].env.OBS_ELEMENT_INDICES[element]
                                first_flag = 0
                                for final_index in element_idx:
                                    if first_flag:
                                        goal_lower[final_index] = -np.array(max_u)[final_index]
                                        break
                                    if np.sign(achieved_goal_reference[final_index]) == -np.sign(np.array(max_u)[final_index]):
                                        first_flag = 1
                                        goal_lower[final_index] = -np.array(max_u)[final_index]

                            obs_temp = goal_lower.copy()
                            achieved_goal_temp = np.zeros_like(obs_temp)
                            for element in self.envs[i].env.TASK_ELEMENTS:
                                element_idx = self.envs[i].env.OBS_ELEMENT_INDICES[element]
                                element_goal = obs_temp[element_idx]
                                achieved_goal_temp[element_idx] = element_goal
                            self.goal_array[hrl_layer_num-1] = achieved_goal_temp.reshape(1,-1).copy()
                        if 'kitchen' in str(self.envs[0]):
                            pass
                        elif 'Rope' in str(self.envs[0]):
                            for num in range(5,20):
                                self.envs[i].env.set_subgoal('subgoal_'+str(num), np.array([u[i][2*(num-5)], u[i][2*(num-5)+1], 0.42]))
                        else:
                            name = 'subgoal_' + str(hrl_layer_num)
                            self.envs[i].env.set_subgoal(name, u[i])
                        self.subgoals_array.append(u[i])
                        curr_obs, curr_ag, lowest_ag, info, lower_rewards = self.rec_rollout_generator(hrl_layer_num - 1, seed)
                        hrl_obs_2 = curr_obs.copy()
                        hrl_ag_2 = curr_ag.copy()
                        infoG = info.copy()

                        # Computing reward(high negative if next lower level is unable to achieve predicted goal)
                        if self.raps and ('Bin' in str(self.envs[0]) or 'Pick' in str(self.envs[0]) or 'Hollow' in str(self.envs[0])):
                            goal_array_raps = self.goal_array[hrl_layer_num - 1][0][:3].reshape(1,-1).copy()
                            lower_reward = self.envs[i].env.compute_reward(curr_ag, goal_array_raps, reward_type='sparse')
                        else:
                            lower_reward = self.envs[i].env.compute_reward(curr_ag, self.goal_array[hrl_layer_num - 1], reward_type='sparse')

                        if self.hac:
                            reward_hac2[i] = -self.T
                            action_temp1 = (hrl_ag[0] - action_offset) / [x for x in max_u]
                            if 'kitchen' in str(self.envs[0]):
                                action_hac1 = action_temp1.copy()
                            else:
                                action_hac1 = np.zeros(4)
                                action_hac1[:3] = action_temp1

                        if lower_reward == 0:
                            final_upper_success = 1
                        else:
                            final_upper_success = 0

                        # Populating predictBuffer
                        if self.predictor_loss and final_upper_success:
                            obs_2.append(hrl_obs.copy())
                            achieved_goals_2.append(hrl_ag.copy())
                            successes_2.append(success.copy())
                            acts_2.append(unscaled_u.copy())
                            saveGoal = self.g.copy()
                            saveGoal[0] = self.goal_array[hrl_layer_num].copy()
                            goals_2.append(saveGoal.copy())
                            env_indexes_2.append([[env_index]])
                            rewards_2.append(reward.copy())

                            obs_2.append(hrl_obs_2.copy())
                            achieved_goals_2.append(hrl_ag_2.copy())
                            if self.testingNow != 1:
                                episode_2 = dict(o=obs_2,
                                           u=acts_2,
                                           g=goals_2,
                                           r=rewards_2,
                                           ag=achieved_goals_2,
                                           env_indexes=env_indexes_2)
                                episode_2 = convert_episode_to_batch_major(episode_2)
                                self.policyList[hrl_layer_num].store_episode_predict(episode_2)

                            obs_2, achieved_goals_2, acts_2, goals_2, successes_2, rewards_2, env_indexes_2 = [], [], [], [], [], [], []

                        if isinstance(lower_reward, list):
                            rew_temp_1 = lower_reward[0]
                        else:
                            rew_temp_1 = lower_reward
                    
                        # Computing success with the last level achieved goal
                        last_reward = self.envs[i].env.compute_reward(lowest_ag, self.goal_array[hrl_layer_num], reward_type='sparse')
                        if isinstance(last_reward, list):
                            rew_temp = last_reward[0]
                        else:
                            rew_temp = last_reward
                        if rew_temp == 0:
                            success[i] = 1

                        if self.hier:
                            reward[i] = np.mean(np.array(lower_rewards), axis=0)[0]
                        else:
                            reward[i] = last_reward
                        # if 0:#self.reward_model:
                        #     reward[i] = self.policyList[hrl_layer_num-1].get_Q_sac_no_ag(hrl_obs[0], self.goal_array[hrl_layer_num-1])[0][0]
                        
                        reward_book[i] = last_reward

                        if t == self.T-1:
                            self.dist_values.append(self.envs[i].env.compute_reward(lowest_ag, self.goal_array[hrl_layer_num], reward_type='dense'))

                        if t == self.T-1:
                            if rew_temp == 0:
                                self.is_successful_values.append(1)
                            else:
                                self.is_successful_values.append(0)

                        for idx, key in enumerate(self.info_keys):
                            info_values[idx][t, i] = info[key]
                        if self.render:
                            if 'kitchen' in str(self.envs[0]):
                                self.envs[i].mj_render()
                            else:
                                self.envs[i].render()
                    except MujocoException as e:
                        return self.rec_rollout_generator()

                if np.isnan(hrl_obs_2).any():
                    self.logger.warn('NaN caught during rollout generation. Trying again...')
                    self.reset_all_rollouts()
                    return self.rec_rollout_generator()

                obs.append(hrl_obs.copy())
                achieved_goals.append(hrl_ag.copy())
                successes.append(success.copy())
                successes_non_contact_walls.append(success_non_contact_walls.copy())
                acts.append(unscaled_u.copy())
                saveGoal = self.g.copy()
                saveGoal[0] = self.goal_array[hrl_layer_num].copy()
                goals.append(saveGoal.copy())
                env_indexes.append([[env_index]])
                rewards.append(reward.copy())
                rewards_book.append(reward_book[0].copy())
                if self.hac:
                    rewards_hac2.append(reward_hac2.copy())
                    achieved_goals_hac1.append([action_hac1])

                hrl_obs[...] = hrl_obs_2.copy()
                hrl_ag[...] = hrl_ag_2.copy()

                import cv2
                self.envs[0].env.render()
                if 'kitchen' in str(self.envs[0]):
                    image = self.envs[0].env.render(mode='rgb_array')
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)[:,:]
                    cv2.imwrite('./figs_1/kitchen/rpl/seed_2/image_'+str(self.subgoals_count)+'.jpg', image)
                    self.subgoals_count += 1
                    print(self.subgoals_count)
                else:    
                    image = self.envs[0].env.capture()
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)[:,:]
                    image = image[200:1000,500:1450]
                    cv2.imwrite('./figs_1/hollow/crisp/seed_0/image_'+str(self.subgoals_count)+'.jpg', image)
                    self.subgoals_count += 1

            obs.append(hrl_obs.copy())
            achieved_goals.append(hrl_ag.copy())

            self.hrl_initial_obs[hrl_layer_num] = hrl_obs.copy()
            self.hrl_initial_ag[hrl_layer_num] = hrl_ag.copy()

            episode = dict(o=obs,
                           u=acts,
                           g=goals,
                           r=rewards,
                           ag=achieved_goals,
                           full_o=self.obs_complete.copy(),
                           env_indexes=env_indexes)

            if self.hac == 1:
                episode_hac1 = dict(o=obs,
                           u=achieved_goals_hac1,
                           g=goals,
                           r=rewards,
                           ag=achieved_goals,
                           full_o=self.obs_complete.copy(),
                           env_indexes=env_indexes)
                episode_hac1 = convert_episode_to_batch_major(episode_hac1)
                if self.testingNow != 1:
                    self.policyList[hrl_layer_num].store_episode(episode_hac1)

            if self.hac and epoch_num % self.hac_iterate == 0:
                episode_hac2 = dict(o=obs,
                           u=acts,
                           g=goals,
                           r=rewards_hac2,
                           ag=achieved_goals,
                           full_o=self.obs_complete.copy(),
                           env_indexes=env_indexes)
                episode_hac2 = convert_episode_to_batch_major(episode_hac2)
                if self.testingNow != 1:
                    self.policyList[hrl_layer_num].store_episode(episode_hac2)

            self.obs_complete, self.achieved_goals_complete, self.acts_complete, self.goals_complete, self.successes_complete, self.rewards_complete, self.env_indexes_complete = [], [], [], [], [], [], []

            for key, value in zip(self.info_keys, info_values):
                episode['info_{}'.format(key)] = value
            
            episode = convert_episode_to_batch_major(episode)
            if self.testingNow != 1:
                self.policyList[hrl_layer_num].store_episode(episode)
            
            # stats
            successful = np.array(successes)[-1, :]
            reward_list = np.array(rewards_book)
            assert successful.shape == (self.rollout_batch_size,)
            success_rate = np.mean(successful)
            average_reward = np.sum(reward_list)
            self.success_history.append(success_rate)
            self.r_upper_history.append(average_reward)
            self.n_episodes += self.rollout_batch_size
            
            return hrl_obs, hrl_ag, lowest_ag, infoG
        else:
            # Lower level policy prediction
            # generate episodes
            obs, achieved_goals, acts, goals, goalsOptimal, successes, rewards, env_indexes, rewards_book = [], [], [], [], [], [], [], [], []
            if self.num_hrl_layers == 1:
                successes_non_contact_walls = []
            info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
            infoG = {}
            num_eps_collisions_with_wall = 0
            hrl_obs = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
            hrl_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals

            hrl_obs = self.hrl_initial_obs[hrl_layer_num].copy()
            hrl_ag = self.hrl_initial_ag[hrl_layer_num].copy()

            obs_test_list = []
            acs_test_list = []
            for t in range(self.T):
                if not self.testingNow:
                    self.current_timestep += 1
                self.curr_eps_step_count += 1
                if self.raps:
                    if 'Rope' in str(self.envs[0]):
                        print('No RAPS baseline for Rope env yet!')
                        assert False
                    elif 'Bin' in str(self.envs[0]) or 'Pick' in str(self.envs[0]) or 'Hollow' in str(self.envs[0]):
                        select_flag = self.goal_array[hrl_layer_num][0][3] 
                        goal_temp1 = self.goal_array[hrl_layer_num][0][:3].reshape(1,-1).copy()
                        if select_flag > -0.1 and select_flag < 0.1:
                            policy_output = self.policyList[hrl_layer_num].get_actions_controller_pick_goTo(hrl_obs[0], hrl_ag, goal_temp1)
                        elif select_flag < 0:
                            policy_output = self.policyList[hrl_layer_num].get_actions_controller_pick_closeGripper(hrl_obs[0], hrl_ag)
                        else:#elif select_flag > 0:
                            policy_output = self.policyList[hrl_layer_num].get_actions_controller_pick_openGripper(hrl_obs[0], hrl_ag)
                    elif 'Reach' in str(self.envs[0]):
                        goal_temp1 = self.goal_array[hrl_layer_num][0][:3].reshape(1,-1).copy()
                        policy_output = self.policyList[hrl_layer_num].get_actions_controller_maze_goTo(hrl_obs[0], hrl_ag, goal_temp1)
                    elif 'kitchen' in str(self.envs[0]):
                        policy_output = self.goal_array[hrl_layer_num][0][:9].reshape(1,-1).copy()
                elif 'Rope' in str(self.envs[0]):
                    policy_output = self.policyList[hrl_layer_num].get_actions_rope_sac(hrl_obs[0], hrl_ag, self.goal_array[hrl_layer_num],
                    compute_Q=self.compute_Q,noise_eps=self.noise_eps if not self.exploit else 0.,
                    random_eps=self.random_eps if not self.exploit else 0.,use_target_net=self.use_target_net, deterministic=self.testingNow)
                    if self.hac:
                        policy_output_2 = self.policyList[hrl_layer_num].get_actions_rope_sac(hrl_obs[0], hrl_ag, self.goal_array[hrl_layer_num],
                        compute_Q=self.compute_Q,noise_eps=self.noise_eps if not self.exploit else 0.,
                        random_eps=self.random_eps if not self.exploit else 0.,use_target_net=self.use_target_net, deterministic=True)
                else:
                    if 1:#self.lower_reward_model:
                        policy_output = self.policyList[hrl_layer_num].get_actions_sac(hrl_obs[0], hrl_ag, self.goal_array[hrl_layer_num],
                        compute_Q=self.compute_Q,noise_eps=self.noise_eps if not self.exploit else 0.,
                        random_eps=self.random_eps if not self.exploit else 0.,use_target_net=self.use_target_net, deterministic=self.testingNow)
                    else:
                        goal_temp1 = self.goal_array[hrl_layer_num][0][:3].reshape(1,-1).copy()
                        policy_output = self.policyList[hrl_layer_num].get_actions_controller_maze_goTo(hrl_obs[0], hrl_ag, goal_temp1)
                    if self.hac:
                        policy_output_2 = self.policyList[hrl_layer_num].get_actions_sac(hrl_obs[0], hrl_ag, self.goal_array[hrl_layer_num],
                        compute_Q=self.compute_Q,noise_eps=self.noise_eps if not self.exploit else 0.,
                        random_eps=self.random_eps if not self.exploit else 0.,use_target_net=self.use_target_net, deterministic=True)

                if self.compute_Q:
                    u, Q = policy_output
                    if self.hac and epoch_num % self.hac_iterate == 0:
                        u, Q = policy_output_2
                    Qs.append(Q)
                else:
                    u = policy_output
                    if self.hac and epoch_num % self.hac_iterate == 0:
                        u = policy_output_2

                if u.ndim == 1:
                    # The non-batched case should still have a reasonable shape.
                    u = u.reshape(1, -1)

                # To limit the movement of the robotic arm
                limit_factor = 1#0.3
                u = u * limit_factor

                reward = np.zeros((self.rollout_batch_size, 1), np.float32)
                reward_final = np.zeros((self.rollout_batch_size, 1), np.float32)
                reward_book = np.zeros((self.rollout_batch_size, 1), np.float32)
                hrl_obs_2 = np.empty((self.rollout_batch_size, self.dims['o']))
                hrl_ag_2 = np.empty((self.rollout_batch_size, self.dims['g']))
                success = np.zeros(self.rollout_batch_size)
                success_non_contact_walls = np.zeros(self.rollout_batch_size)
                # compute new states and observations
                for i in range(self.rollout_batch_size):
                    try:
                        action_n = u[i]
                        acs_test_list.append(action_n)
                        if 'kitchen' in str(self.envs[0]):
                            curr_o_new, reward_n, done, info = self.envs[i].step(action_n)
                        elif 'Rope' in str(self.envs[0]):
                            curr_o_new, reward_n, done, info = self.envs[i].step_rope(action_n)
                        else:
                            curr_o_new, reward_n, done, info = self.envs[i].step_maze(action_n)
                        # print('lower_reward', reward_n)
                        obs_test_list.append(curr_o_new)

                        if num_hrl_layers > 1:
                            final_reward = self.envs[i].env.compute_reward(curr_o_new['achieved_goal'].reshape(1,-1), self.goal_array[hrl_layer_num + 1], reward_type='sparse')
                        else:
                            final_reward = [0]

                        infoG = info.copy()
                        if num_hrl_layers==1 and t == self.T-1:
                            self.dist_values.append(self.envs[i].env.compute_reward(curr_o_new['achieved_goal'].reshape(1,-1), self.goal_array[hrl_layer_num], reward_type='dense'))
                        if self.num_hrl_layers == 1:
                            if reward_n == 0 and self.num_wall_collision <= 4:
                                success_non_contact_walls[i] = 1
                        if reward_n == 0:
                            success[i] = 1
                        
                        if self.hac:
                            reward[i] = reward_n
                        if self.hier:
                            reward[i] = final_reward[0]
                        reward_final[i] = final_reward[0]

                        reward_book[i] = reward_n
                        hrl_obs_2[i] = curr_o_new['observation']

                        hrl_ag_2[i] = curr_o_new['achieved_goal']
                        for idx, key in enumerate(self.info_keys):
                            info_values[idx][t, i] = info[key]
                        if self.render:
                            import time
                            time.sleep(0.01)
                            self.envs[i].render()
                    except MujocoException as e:
                        return self.rec_rollout_generator(self.policyList)

                if np.isnan(hrl_obs_2).any():
                    self.logger.warn('NaN caught during rollout generation. Trying again...')
                    self.reset_all_rollouts()
                    return self.rec_rollout_generator(self.policyList)

                obs.append(hrl_obs.copy())
                achieved_goals.append(hrl_ag.copy())
                successes.append(success.copy())
                acts.append(u.copy())
                saveGoal = self.g.copy()
                if self.raps and ('Bin' in str(self.envs[0]) or 'Pick' in str(self.envs[0]) or 'Hollow' in str(self.envs[0])):
                    saveGoal[0] = self.goal_array[hrl_layer_num][0][:3].reshape(1,-1).copy()
                else:
                    saveGoal[0] = self.goal_array[hrl_layer_num]
                goals.append(saveGoal.copy())
                if num_hrl_layers > 1:
                    saveGoal[0] = self.goal_array[hrl_layer_num + 1]
                    goalsOptimal.append(saveGoal.copy())
                rewards.append(reward.copy())
                env_indexes.append([[env_index]])
                rewards_book.append(reward_book[0].copy())

                if not self.testingNow:
                    self.obs_final.append(hrl_obs.copy())
                    self.achieved_goals_final.append(hrl_ag.copy())
                    self.acts_final.append(u.copy())
                    saveGoal = self.g.copy()
                    if num_hrl_layers > 1:
                        saveGoal[0] = self.goal_array[hrl_layer_num + 1]
                    self.goals_final.append(saveGoal.copy())
                    self.rewards_final.append(reward_final.copy())
                    self.env_indexes_final.append([[env_index]])

                self.obs_complete.append(hrl_obs.copy())
                self.achieved_goals_complete.append(hrl_ag.copy())
                self.acts_complete.append(u.copy())
                saveGoal = self.g.copy()
                if num_hrl_layers > 1:
                    saveGoal[0] = self.goal_array[hrl_layer_num + 1]
                self.goals_complete.append(saveGoal.copy())
                self.rewards_complete.append(reward_final.copy())
                self.env_indexes_complete.append([[env_index]])



                # Appending to global episodes
                self.global_episode['u'].append(hrl_ag_2.copy()[0])
                # Tells how good is the current state, wrt the final goal
                lower_reward = self.envs[0].env.compute_reward(hrl_ag_2, self.goal_array[-1], reward_type='sparse')
                if isinstance(lower_reward, list):
                    rew_temp = lower_reward[0]
                else:
                    rew_temp = lower_reward
                lower_reward = rew_temp
                self.global_episode['success'].append(lower_reward)

                hrl_obs[...] = hrl_obs_2
                hrl_ag[...] = hrl_ag_2

            obs.append(hrl_obs.copy())
            achieved_goals.append(hrl_ag.copy())

            if not self.testingNow:
                self.obs_final.append(hrl_obs.copy())
                self.achieved_goals_final.append(hrl_ag.copy())

            # self.obs_complete.append(hrl_obs.copy())
            # self.achieved_goals_complete.append(hrl_ag.copy())

            self.hrl_initial_obs[hrl_layer_num] = hrl_obs.copy()
            self.hrl_initial_ag[hrl_layer_num] = hrl_ag.copy()
            episode = dict(o=obs,
                           u=acts,
                           g=goals,
                           r=rewards,
                           ag=achieved_goals,
                           env_indexes=env_indexes)

            # Episode with goal as final goal(optimal policy)
            optimalFlag = 0
            episodeOptimal = None
            if len(self.acts_final) == episode_len:
                optimalFlag = 0
                self.current_timestep = -1
            else:
                optimalFlag = 0
            if not self.testingNow and optimalFlag:
                episodeOptimal =  dict(o=self.obs_final.copy(),
                                       u=self.acts_final.copy(),
                                       g=self.goals_final.copy(),
                                       r=self.rewards_final.copy(),
                                       ag=self.achieved_goals_final.copy(),
                                       env_indexes=self.env_indexes_final).copy()
            for key, value in zip(self.info_keys, info_values):
                episode['info_{}'.format(key)] = value
                if optimalFlag:
                    episodeOptimal['info_{}'.format(key)] = value

            # stats
            if self.num_hrl_layers == 1:
                successful = np.array(successes)[-1, :]
                assert successful.shape == (self.rollout_batch_size,)
                success_rate = np.mean(successful)
                self.success_history.append(success_rate)
            reward_list = np.array(rewards_book)
            average_reward = np.sum(reward_list)
            self.r_lower_history.append(average_reward)
            if self.compute_Q:
                self.Q_history.append(np.mean(Qs))
            num_episodes = self.T
            Qs = []
            
            episode = convert_episode_to_batch_major(episode)
            if optimalFlag:
                episodeOptimal = convert_episode_to_batch_major(episodeOptimal)
            if self.testingNow != 1:
                self.policyList[hrl_layer_num].store_episode(episode)
                if optimalFlag:
                    self.policyList[hrl_layer_num].store_episode(episodeOptimal)
                    self.obs_final, self.achieved_goals_final, self.acts_final, self.goals_final, self.successes_final, self.rewards_final, self.env_indexes_final = [], [], [], [], [], [], []
            return hrl_obs.copy(), hrl_ag.copy(), hrl_ag.copy(), infoG, rewards

    def reset_global_buffer(self):
        self.global_episode = dict(u=[], success=[])

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.r_upper_history.clear()
        self.r_lower_history.clear()
        # self.success_history_non_contact_walls.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_average_reward(self):
        return np.mean(self.average_reward)

    def current_success_rate_non_contact_walls(self):
        return np.mean(self.success_history_non_contact_walls)    

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policyList, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        if prefix == 'train':
            logs += [('4.  Train episodes elapsed', self.n_episodes)]
            logs += [('5.  Train success rate', np.mean(self.success_history))]
            logs += [('6.  Train average upper reward', np.mean(self.r_upper_history))]
            logs += [('7.  Train average lower reward', np.mean(self.r_lower_history))]
        elif prefix == 'test':
            logs += [('8.  Test episodes elapsed', self.n_episodes)]
            logs += [('9.  Test success rate', np.mean(self.success_history))]
            logs += [('X.  Test average upper reward', np.mean(self.r_upper_history))]
            logs += [('XI. Test average lower reward', np.mean(self.r_lower_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)

