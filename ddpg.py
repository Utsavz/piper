from collections import OrderedDict
from tensorflow import keras
import numpy as np
import tensorflow as tf
import os
import pickle
from tensorflow.contrib.staging import StagingArea
from baselines import logger
from util import (import_function, store_args, flatten_grads, transitions_in_episode_batch)
from normalizer import Normalizer
from replay_buffer import ReplayBuffer
from memory import Memory
from baselines.common.mpi_adam import MpiAdam
from baselines.her.util import convert_episode_to_batch_major
from actor_critic_sac import ActorCriticSac
import random
import cv2
import time
import gym
from tqdm import tqdm

def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}

global demoBuffer0
global demoBuffer1
global predictBuffer
global stored_hrl_buffer
global stored_adversarial_buffer

class DDPG(object):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size, reward_batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns,  clip_return, bc_loss, q_filter, num_demo,
                 hrl_imitation_loss, adversarial_loss, num_upper_demos, sample_transitions, gamma, alpha, sac, hac, dac, bc_reg, q_reg, hier, 
                 predictor_loss, info, optimal_policy=None, lower_policy=None, reward_model=0, lower_reward_model=0, reuse=False, **kwargs):
        """Implementation of SAC that is used in combination with Hindsight Experience Replay (HER).

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'baselines.her.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per SAC agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
        """
        if self.clip_return is None:
            self.clip_return = np.inf
        self.sac = sac
        self.predictor_loss = predictor_loss
        self.upper_only_bc = 0
        self.is_image_data = 0
        self.is_multiple_env = 1
        self.env_name=info['env_name']
        self.num_hrl_layers = kwargs['num_hrl_layers']
        self.num_upper_demos = num_upper_demos
        self.reward_model = reward_model
        self.lower_reward_model = lower_reward_model
        self.demo_batch_size = 128

        if self.scope == 'sac0':
            if lower_reward_model:
                self.reward_model = 1
                if self.bc_loss:
                    self.demo_batch_size = 10
            else:
                self.reward_model = 0

        if self.reward_model:
            self.batch_size = reward_batch_size

        if self.sac:
            self.create_actor_critic = ActorCriticSac
        else:
            self.create_actor_critic = ActorCritic #import_function(self.network_class)

        self.dac = dac
        self.hac = hac
        self.hier = hier
        self.bc_reg = bc_reg
        self.q_reg = q_reg

        self.margin_loss = 0
        if self.margin_loss:
            self.margin_lambda = 0.0001

        if self.scope != 'sac0' and 'Rope' in self.env_name:
            self.input_dims['u'] = 15
        if self.scope != 'sac0' and 'kitchen' in self.env_name:
            self.input_dims['u'] = 30
        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        if self.upper_only_bc:
            self.demo_batch_size = 1024
        self.adversarial_batch_size = 128

        self.predict_batch_size = 128
        self.adversarial_predict_batch_size = 128
        self.hrl_imitation_batch_size = 128
        self.real_batch_size = 128
        self.lambda1 = 0.001
        self.lambda2 =  0.0078

        if 'Pick' in self.env_name:
            self.lambda2 =  0.005
        if 'Maze' in self.env_name:
            self.lambda2 =  0.001
        if 'Rope' in self.env_name:
            self.lambda2 =  0.005
        if 'kitchen' in self.env_name:
            self.lambda2 =  0.001
        
        self.l2_reg_coeff = 0.005
        self.image_dim = 30
        self.image_size = self.image_dim * self.image_dim #* 3
        self.max_u = max_u
        self.stored_buffer = 0
        self.stored_ad_buffer = 0
        if self.stored_ad_buffer:
            global stored_adversarial_buffer
            self.success_buffer = stored_adversarial_buffer
        else:
            self.success_buffer = {}
        self.sg_baseline_count = 0
        self.env = gym.make(self.env_name)
        self.env.reset()
        self.traj_len = self.T
        if 'Rope' in self.env_name:
            self.T = 5
        else:
            self.T = 10
        self.logdir = ''
        self.otherPolicy = None
        self.optimal_policy = optimal_policy
        self.lower_policy = lower_policy

        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        if self.is_image_data:
            stage_shapes['p'] = (None,3)
            stage_shapes['p_2'] = (None,3)    
        stage_shapes['r'] = (None,)
        self.stage_shapes = stage_shapes
        self.image_shapes = self.stage_shapes.copy()

        if self.reward_model:
            self.stage_shapes['reward_o_1'] = (None, self.traj_len, *input_shapes['o'])
            self.stage_shapes['reward_o_2'] = (None, self.traj_len, *input_shapes['o'])
            self.stage_shapes['reward_g_1'] = (None, self.traj_len, *input_shapes['g'])
            self.stage_shapes['reward_g_2'] = (None, self.traj_len, *input_shapes['g'])
            self.stage_shapes['reward_u_1'] = (None, self.traj_len, *input_shapes['u'])
            self.stage_shapes['reward_u_2'] = (None, self.traj_len, *input_shapes['u'])
            self.stage_shapes['reward_r_1'] = (None, 1)
            self.stage_shapes['reward_r_2'] = (None, 1)

        # Create network.
        if self.is_image_data:
            self.stage_shapes['o'] = (None, self.image_size)
            self.stage_shapes['o_2'] = (None, self.image_size)
            self.image_shapes['o'] = (None, self.image_size)
            self.image_shapes['o_2'] = (None, self.image_size)
            self.image_shapes['p'] = (None,3)
            self.image_shapes['p_2'] = (None,3)
        with tf.compat.v1.variable_scope(self.scope):
            if self.is_image_data:
                self.staging_tf = StagingArea(
                    dtypes=[tf.float32 for _ in self.image_shapes.keys()],
                    shapes=list(self.image_shapes.values()))
                self.buffer_ph_tf = [
                    tf.placeholder(tf.float32, shape=shape) for shape in self.image_shapes.values()]
            else:
                self.staging_tf = StagingArea(
                    dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                    shapes=list(self.stage_shapes.values()))
                self.buffer_ph_tf = [
                    tf.compat.v1.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(reuse=reuse)

        self.min_index = 0
        self.min_dist = 0

        # Configure the replay buffer.
        buffer_shapes = {key: (self.T if key != 'o' else self.T+1, *input_shapes[key])
                         for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T+1, self.dimg)
        buffer_shapes['r'] = (buffer_shapes['g'][0], 1)
        if self.scope != "sac0":
            buffer_shapes['full_o'] = (buffer_shapes['g'][0], self.dimo)

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions, self.optimal_policy, self.lower_policy)

        if 'Rope' in self.env_name:
            self.T = 5
        else:
            self.T = 250
        buffer_shapes = {key: (self.T if key != 'o' else self.T+1, *input_shapes[key])
                         for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T+1, self.dimg)
        buffer_shapes['r'] = (buffer_shapes['g'][0], 1)


        if self.bc_loss or self.bc_loss_upper:
            if self.scope == 'sac0':
                global demoBuffer0
                demoBuffer0 = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions, self.optimal_policy, self.lower_policy)
            else:
                global demoBuffer1
                demoBuffer1 = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions, self.optimal_policy, self.lower_policy)
            if self.scope != 'sac0' and self.predictor_loss:
                global predictBuffer
                predictBuffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions, self.optimal_policy, self.lower_policy)

        self.decimal_num = 2

    def get_demoBuffer_data(self):
        '''
            Get data from lower demonstration replay buffer
        '''
        global demoBuffer0
        return demoBuffer0.get_data()

    def set_adversarial_loss(self):
        '''
        Sets adversarial loss parameter
        '''
        self.adversarial_loss = 1

    def _random_action(self, n):
        '''
        Sample random action of input batch size n
        '''
        return np.random.uniform(low=-1., high=1., size=(n, self.dimu))

    def _preprocess_og(self, o, ag, g):
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def _preprocess_og_no_ag(self, o, g):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def _preprocess_og_no_ag_reward(self, o):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        return o

    def get_actions_controller_maze_goTo(self, o, ag, g):
        # Action controller for maze environment for input batch of observations, achieved goals and final goal
        o, g = self._preprocess_og(o, ag, g)
        gripperPos = o[:3]
        action = [0, 0, 0, 0]
        object_rel_pos = g-gripperPos
        object_oriented_goal = object_rel_pos.copy()

        for i in range(len(object_oriented_goal[0])):
            action[i] = object_oriented_goal[0][i]*6

        action[len(action)-1] = 0.05
        ret = np.array(action)

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def get_actions_controller_pick_goTo(self, o, ag, g):
        # goTo: Action controller for pick and place environment for input batch of observations, achieved goals and final goal
        o, g = self._preprocess_og(o, ag, g)
        gripperPos = o[:3]
        action = [0, 0, 0, 0]
        object_rel_pos = g-gripperPos
        object_oriented_goal = object_rel_pos.copy()

        for i in range(len(object_oriented_goal[0])):
            action[i] = object_oriented_goal[0][i]*6

        action[len(action)-1] = 0.05
        ret = np.array(action)

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def get_actions_controller_pick_closeGripper(self, o, ag):
        # closeGripper: Action controller for pick and place environment for input batch of observations, achieved goals
        action = [0, 0, 0, 0]
        action[len(action)-1] = -0.05
        ret = np.array(action)

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def get_actions_controller_pick_openGripper(self, o, ag):
        # openGripper: Action controller for pick and place environment for input batch of observations, achieved goals
        action = [0, 0, 0, 0]

        action[len(action)-1] = 0.05
        ret = np.array(action)

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def get_actions_controller_rope(self, env, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False, deterministic=False):
        # Action controller for rope environment for input batch of observations, achieved goals and final goal
        o, g = self._preprocess_og(o, ag, g)
        block_reward = self.compute_reward(o, g[0])
        if block_reward == 0:
            return np.array([-1, -1, 0, 0])
        o = o.copy()
        g = g.copy()
        goal_obs_demo = g[0]
        action_final = np.zeros(4) 
        action_start = np.zeros(3)
        action_start[2] = 0.42
        rope_size = 15
        goal_pos = []
        obs_start = []
        curr_pos = []
        num = 0
        selected_num = 4
        curr_dist = -1
        for num in range(4, rope_size):
            goal_pos = goal_obs_demo[2*num:2*(num+1)]
            curr_pos = o[2*num:2*(num+1)]
            curr_selected_dist = self.goal_distance(curr_pos, goal_pos)
            if curr_selected_dist > curr_dist:
                curr_dist =  curr_selected_dist
                selected_num = num
        goal_pos = goal_obs_demo[2*selected_num:2*(selected_num+1)]
        curr_pos = o[2*selected_num:2*(selected_num+1)]

        action_start[:2] = curr_pos[:2].copy()
        x2, y2 = ( goal_pos[0], goal_pos[1])
        x1, y1 = ( curr_pos[0], curr_pos[1])

        slope = (y2-y1) / (x2-x1+0.000001)

        max_drag_dist = 0.08
        angle_radian = np.arctan(slope)
        drag_dist_1 = self.goal_distance(action_start[:2].copy(), goal_pos[:2].copy())
        dist_val = max_drag_dist - drag_dist_1
        if dist_val < 0:
            dist_val = max_drag_dist/2.

        action_start_1 = action_start.copy()
        action_start_2 = action_start.copy()
        action_start_1[0] += dist_val*np.cos(angle_radian)
        action_start_1[1] += dist_val*np.sin(angle_radian)
        drag_dist_1 = self.goal_distance(action_start_1[:2].copy(), goal_pos[:2].copy())
        action_start_2[0] -= dist_val*np.cos(angle_radian)
        action_start_2[1] -= dist_val*np.sin(angle_radian)
        drag_dist_2 = self.goal_distance(action_start_2[:2].copy(), goal_pos[:2].copy())
        drag_sign = 1
        x_min = 1.05
        x_max = 1.55
        y_min = 0.48
        y_max = 1.02
        if drag_dist_2 < drag_dist_1:
            drag_sign = 1
            action_start[0] = action_start_1[0]
            action_start[1] = action_start_1[1]
        else:
            drag_sign = -1
            action_start[0] = action_start_2[0]
            action_start[1] = action_start_2[1]

        
        drag_angle_1 = np.degrees(angle_radian) + 180
        action_x = action_start[0] + max_drag_dist*np.cos(np.radians(drag_angle_1))*1.0
        action_y = action_start[1] + max_drag_dist*np.sin(np.radians(drag_angle_1))*1.0
        action_x = max(x_min, action_x)
        action_x = min(x_max, action_x)
        action_y = max(y_min, action_y)
        action_y = min(y_max, action_y)
        action_end_1 = np.array([action_x, action_y, 0.5])

        drag_angle_2 = np.degrees(angle_radian)
        if drag_angle_2 < 0:
            drag_angle_2 = drag_angle_2 + 360
        action_x = action_start[0] + max_drag_dist*np.cos(np.radians(drag_angle_2))*1.0
        action_y = action_start[1] + max_drag_dist*np.sin(np.radians(drag_angle_2))*1.0
        action_x = max(x_min, action_x)
        action_x = min(x_max, action_x)
        action_y = max(y_min, action_y)
        action_y = min(y_max, action_y)
        action_end_2 = np.array([action_x, action_y, 0.5])

        goal_dist_1 = self.goal_distance(action_end_1[:2].copy(), goal_pos[:2].copy())
        goal_dist_2 = self.goal_distance(action_end_2[:2].copy(), goal_pos[:2].copy())
        if goal_dist_1 < goal_dist_2:
            drag_angle = drag_angle_1
        else:
            drag_angle = drag_angle_2

        drag_angle_norm = (drag_angle-180)/180.0

        action_start[0] = max(1.05, action_start[0])
        action_start[0] = min(1.55, action_start[0])
        action_start[1] = max(0.48, action_start[1])
        action_start[1] = min(1.02, action_start[1])

        iteration = 0
        for num in range(rope_size):
            iteration += 1
            curr_pos = o[2*num:2*(num+1)]
            curr_dist = np.linalg.norm(action_start[:2] - curr_pos[:2])
            if curr_dist < 0.03:
                action_start[0] = action_start[0] + drag_sign*0.03*np.cos(np.arctan(slope))
                action_start[1] = action_start[1] + drag_sign*0.03*np.sin(np.arctan(slope))
                action_start[0] = max(1.05, action_start[0])
                action_start[0] = min(1.55, action_start[0])
                action_start[1] = max(0.48, action_start[1])
                action_start[1] = min(1.02, action_start[1])
            if iteration == rope_size:
                break

        max_u = [0.25, 0.27, 0.145]
        action_offset = [1.3, 0.75, 0.555]
        action_start1 = (action_start - action_offset) / [x for x in max_u]

        action_final[0] = action_start1[0]
        action_final[1] = action_start1[1]
        action_final[2] = drag_angle_norm
        action_final[3] = 0.
        ret = np.array([action_final])

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def get_actions(self, o, ag, g, current_transition, demoData_obs_temp_train, env_index = 0, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
        # Returns actions for input batch of observations, achieved goals and final goal
        o, g = self._preprocess_og(o, ag, g)

        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.Q_pi_tf]
        # feed
        if self.is_image_data:
            o_pixel = o[3:].reshape(-1, self.image_size)
        else:
            o_pixel = o.reshape(-1, self.dimo)
        if self.is_image_data:
            feed = {
                policy.o_tf: o_pixel,
                policy.g_tf: g.reshape(-1, self.dimg),
                policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32),
                policy.p_tf: o[:3].reshape(-1, 3),
            }
        else:
            feed = {
            policy.o_tf: o_pixel,
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32),
        }
        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        if not self.discrete_maze:
            # pass
            nn = np.random.randn(*u.shape)
            noise = noise_eps * nn  # gaussian noise
            u += noise
            if 'Rope' not in self.env_name:
                add_excess = np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
                u += add_excess
        else:
            COMPASS = {
                    "N": (0, -1),
                    "E": (1, 0),
                    "S": (0, 1),
                    "W": (-1, 0),
                    "NOP": (0,0),
                }
            COMPASS_INDEX = ["N","E","S","W","NOP"]
            if np.random.rand() < 0.5:
                index = COMPASS_INDEX[np.random.randint(5)]
                u += COMPASS[index]
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def get_actions_rope_sac(self, o, ag, g, env_index = 0, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False, deterministic=False):
        # get actions for rope environment for input batch of observations, achieved goals and final goal
        o, g = self._preprocess_og(o, ag, g)
        if 'Rope' in self.env_name:
            block_reward = self.compute_reward(o, g[0])
            if block_reward == 0:
                return np.array([-1, -1, 0, 0])
        policy = self.target if use_target_net else self.main
        # values to compute
        if deterministic:
            vals = [policy.mu_tf]
        else:    
            vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.Q1_pi_tf]
        # feed
        if self.is_image_data:
            o_pixel = self.get_image_obs(o, g[0], [env_index])
            o_pixel = o_pixel.reshape(-1, self.image_size)
        else:
            o_pixel = o.reshape(-1, self.dimo)
        if self.is_image_data:
            feed = {
                policy.o_tf: o_pixel,
                policy.g_tf: g.reshape(-1, self.dimg),
                policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32),
                policy.p_tf: o[:3].reshape(-1, 3),
            }
        else:
            feed = {
                policy.o_tf: o_pixel,
                policy.g_tf: g.reshape(-1, self.dimg),
                policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32),
            }

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret


    def get_actions_sac(self, o, ag, g, env_index = 0, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False, deterministic=False):
        # get actions for pick and place, bin, hollow etc environments for input batch of observations, achieved goals and final goal
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target if use_target_net else self.main
        # values to compute
        if deterministic:
            vals = [policy.mu_tf]
        else:    
            vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.Q1_pi_tf]
        # feed
        if self.is_image_data:
            o_pixel = self.get_image_obs(o, g[0], [env_index])
            o_pixel = o_pixel.reshape(-1, self.image_size)
        else:
            o_pixel = o.reshape(-1, self.dimo)
        if self.is_image_data:
            feed = {
                policy.o_tf: o_pixel,
                policy.g_tf: g.reshape(-1, self.dimg),
                policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32),
                policy.p_tf: o[:3].reshape(-1, 3),
            }
        else:
            feed = {
                policy.o_tf: o_pixel,
                policy.g_tf: g.reshape(-1, self.dimg),
                policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32),
            }

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret


    def get_Q_sac(self, o, ag, g, env_index = 0, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False, deterministic=False):
        # get Q values for various environment for input batch of observations, achieved goals and final goal
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.Q1_pi_tf]
        # feed
        if self.is_image_data:
            o_pixel = self.get_image_obs(o, g[0], [env_index])
            o_pixel = o_pixel.reshape(-1, self.image_size)
        else:
            o_pixel = o.reshape(-1, self.dimo)
        if self.is_image_data:
            feed = {
                policy.o_tf: o_pixel,
                policy.g_tf: g.reshape(-1, self.dimg),
                policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32),
                policy.p_tf: o[:3].reshape(-1, 3),
            }
        else:
            feed = {
                policy.o_tf: o_pixel,
                policy.g_tf: g.reshape(-1, self.dimg),
                policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32),
            }

        ret = self.sess.run(vals, feed_dict=feed)
        return ret

    def get_Q_sac_reward_upper(self, o, g, u, env_index = 0, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False, deterministic=False):
        # get Q values for various environment for input batch of observations, achieved goals and final goal
        o, g = self._preprocess_og_no_ag(o, g)
        policy = self.target if use_target_net else self.main
        ret_shape = np.array(o).shape
        # values to compute
        vals = [policy.Q1_tf]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: u.reshape(-1, self.dimu),
        }
        ret = self.sess.run(vals, feed_dict=feed)
        ret = np.array(ret)
        ret = ret.reshape(ret_shape[0], ret_shape[1], 1)
        return ret

    def get_Q_sac_reward_lower(self, o, g, env_index = 0, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False, deterministic=False):
        # get Q values for various environment for input batch of observations, achieved goals and final goal
        o, g = self._preprocess_og_no_ag(o, g)
        policy = self.target if use_target_net else self.main
        ret_shape = np.array(o).shape
        # values to compute
        vals = [policy.Q1_pi_tf]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32),
        }
        ret = self.sess.run(vals, feed_dict=feed)
        ret = np.array(ret)
        ret = ret.reshape(ret_shape[0], ret_shape[1], 1)
        return ret

    def get_Q_sac_no_ag(self, o, g, env_index = 0, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False, deterministic=False):
        # get Q values for various environment for input batch of observations, achieved goals and final goal
        o, g = self._preprocess_og_no_ag(o, g)
        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.Q1_pi_tf]
        # feed
        if self.is_image_data:
            o_pixel = self.get_image_obs(o, g[0], [env_index])
            o_pixel = o_pixel.reshape(-1, self.image_size)
        else:
            o_pixel = o.reshape(-1, self.dimo)
        if self.is_image_data:
            feed = {
                policy.o_tf: o_pixel,
                policy.g_tf: g.reshape(-1, self.dimg),
                policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32),
                policy.p_tf: o[:3].reshape(-1, 3),
            }
        else:
            feed = {
                policy.o_tf: o_pixel,
                policy.g_tf: g.reshape(-1, self.dimg),
                policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32),
            }

        ret = self.sess.run(vals, feed_dict=feed)
        return ret

    def get_actions_sac_no_ag(self, o, g, env_index = 0, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False, deterministic=False):
        o, g = self._preprocess_og_no_ag(o, g)
        policy = self.target if use_target_net else self.main
        # values to compute
        if deterministic:
            vals = [policy.mu_tf]
        else:    
            vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.Q1_pi_tf, policy.Q1_pi_tf]
        # feed
        if self.is_image_data:
            o_pixel = self.get_image_obs(o, g[0], [env_index])
            o_pixel = o_pixel.reshape(-1, self.image_size)
        else:
            o_pixel = o.reshape(-1, self.dimo)
        if self.is_image_data:
            feed = {
                policy.o_tf: o_pixel,
                policy.g_tf: g.reshape(-1, self.dimg),
                policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32),
                policy.p_tf: o[:3].reshape(-1, 3),
            }
        else:
            feed = {
                policy.o_tf: o_pixel,
                policy.g_tf: g.reshape(-1, self.dimg),
                policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32),
            }

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret


    def get_rewards_sac_no_ag(self, o, g, u, env_index = 0, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False, deterministic=False):
        o, g = self._preprocess_og_no_ag(o, g)
        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [self.main.reward_pred_tf]
        # feed
        feed = {
                policy.o_tf: o.reshape(-1, self.dimo),
                policy.g_tf: g.reshape(-1, self.dimg),
                policy.u_tf: u.reshape(-1, self.dimu),
            }

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret



    def get_discriminator_output(self, o, g, u):
        # get discriminator outputs for various environments for input batch of observations, achieved goals and actions
        o, g = self._preprocess_og_no_ag(o, g)
        o_pixel = o.reshape(-1, self.dimo)
        g_pixel = g.reshape(-1, self.dimg)
        u_pixel = u.reshape(-1, self.dimu)
        policy = self.main
        # values to compute
        vals = [policy.discriminator_pred_tf]
        # feed
        feed = {
            policy.o_tf: o_pixel,
            policy.g_tf: g_pixel,
            policy.u_tf: u_pixel,
        }

        ret = self.sess.run(vals, feed_dict=feed)
        u = ret[0]
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def get_predictor_output(self, o, g, u):
        # get predictor outputs for various environments
        o, g = self._preprocess_og_no_ag(o, g)
        o_pixel = o.reshape(-1, self.dimo)
        g_pixel = g.reshape(-1, self.dimg)
        u_pixel = u.reshape(-1, self.dimu)
        policy = self.main
        # values to compute
        vals = [policy.predictor_pred_tf]
        # feed
        feed = {
            policy.o_tf: o_pixel,
            policy.g_tf: g_pixel,
            policy.u_tf: u_pixel,
        }

        ret = self.sess.run(vals, feed_dict=feed)
        u = ret[0]
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def initDemoBuffer(self, demoDataFile, update_stats=True, limit_data=-1):
        # Initialize lower level expert demonstration buffer
        print('Lower level data loading from', demoDataFile)
        demoData = np.load(demoDataFile, allow_pickle=True)
        total_buffer_size = 0
        obs_1_list = demoData['obs']
        acs_1_list = demoData['acs']
        self.update_stats_counter = 0
        del demoData
        if 'Rope' in self.env_name:
            max_u = np.array([])
            action_offset = np.array([])
            for i in range(15):
                max_u = np.append(max_u, [0.25, 0.27], axis=0)
            for i in range(15):
                action_offset = np.append(action_offset, [1.3, 0.75], axis=0)
        elif 'kitchen' in self.env_name:
            max_u = [1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,0.88,0.01,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,0.75,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6]
            action_offset = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        else:
            max_u = [0.25, 0.27, 0.145]
            action_offset = [1.3, 0.75, 0.555]
        transition_buffer_size = 0
        num_env = self.num_upper_demos
        # if limit_data != -1:
        #     num_env = limit_data
        # else:
        #     num_env = len(acs_1_list)
        for_num_env = range(num_env)
        for env_index in tqdm(for_num_env):
            if 'Maze' in self.env_name:
                self.env.env.setIndex(env_index)
                maze_array = self.env.env.get_maze_array().ravel().tolist().copy()
                gates = self.env.env.gates.copy()
            if np.array(obs_1_list[env_index]).shape[0] == 0:
                continue
            if 'Maze' not in self.env_name:
                obs_1 = obs_1_list[env_index].copy()
                acs_1 = acs_1_list[env_index].copy()
            else:
                obs_1 = obs_1_list[env_index][0].copy()
                acs_1 = acs_1_list[env_index][0].copy()
            obs, acts, goals, achieved_goals , rews, env_indexes = [], [] ,[] ,[], [], []
            i = 0
            ep_len = len(acs_1)
            for transition in range(ep_len):
                total_buffer_size += 1
                obs_temp = obs_1[transition].get('observation').copy()
                if 'Maze' in self.env_name:
                    obs_temp = np.array(obs_temp[:3].tolist() + maze_array.copy())
                obs.append(np.array([obs_temp.copy()]))
                acts.append(np.array([acs_1[transition].copy()]))
                rews.append(np.array([0]))#rew_1[env_index][transition]])
                goals.append(np.array([obs_1[transition].get('desired_goal').copy()]))
                achieved_goals.append(np.array([obs_1[transition].get('achieved_goal')].copy()))
                env_indexes.append(np.array([np.array([env_index])].copy()))
                rew = []

            obs_temp = obs_1[-1].get('observation').copy()
            if 'Maze' in self.env_name:
                obs_temp = np.array(obs_temp[:3].tolist() + maze_array.copy())
            obs.append([obs_temp.copy()])
            achieved_goals.append([obs_1[-1].get('achieved_goal').copy()])

            episode = dict(o=obs,
                           u=acts,
                           g=goals,
                           r=rews,
                           ag=achieved_goals,
                            env_indexes=env_indexes)

            episode = convert_episode_to_batch_major(episode)
            demoBuffer0.store_episode(episode)

            if update_stats and self.update_stats_counter < 500:
                self.update_stats_counter += 1
                # add transitions to normalizer to normalize the demo data as well
                episode['o_2'] = episode['o'][:, 1:, :]
                episode['ag_2'] = episode['ag'][:, 1:, :]
                num_normalizing_transitions = transitions_in_episode_batch(episode)
                transitions = self.sample_transitions(episode, num_normalizing_transitions)

                o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
                transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
                
                # No need to preprocess the o_2 and g_2 since this is only used for stats
                self.o_stats.update(transitions['o'])
                self.g_stats.update(transitions['g'])

                self.o_stats.recompute_stats()
                self.g_stats.recompute_stats()
            episode.clear()
        del obs_1_list
        del acs_1_list
        print("Lower level demo buffer loaded. Size: ", total_buffer_size)


    def initDemoBufferPredict(self, demoDataFile, update_stats=True):
        # Initialize lower level predict expert demonstration buffer
        print('Predict buffer data loading...')
        demoData = np.load(demoDataFile, allow_pickle=True)
        total_buffer_size = 0
        obs_1_list = demoData['obs']
        acs_1_list = demoData['acs']
        del demoData
        if 'Rope' in self.env_name:
            max_u = np.array([])
            action_offset = np.array([])
            for i in range(15):
                max_u = np.append(max_u, [0.25, 0.27], axis=0)
            for i in range(15):
                action_offset = np.append(action_offset, [1.3, 0.75], axis=0)
        elif 'kitchen' in self.env_name:
            max_u = [1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,0.88,0.01,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,0.75,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6]
            action_offset = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        else:
            max_u = [0.25, 0.27, 0.145]
            action_offset = [1.3, 0.75, 0.555]
        transition_buffer_size = 0
        num_env = self.num_upper_demos
        for_num_env = range(num_env)
        for env_index in tqdm(for_num_env):
            self.env.env.setIndex(env_index)
            if 'Maze' in self.env_name:
                maze_array = self.env.env.get_maze_array().ravel().tolist().copy()
                gates = self.env.env.gates.copy()
            if np.array(obs_1_list[env_index]).shape[0] == 0:
                continue
            if 'Maze' not in self.env_name:
                obs_1 = obs_1_list[env_index].copy()
                acs_1 = acs_1_list[env_index].copy()
            else:
                obs_1 = obs_1_list[env_index][0].copy()
                acs_1 = acs_1_list[env_index][0].copy()
            obs, acts, goals, achieved_goals , rews, env_indexes = [], [] ,[] ,[], [], []
            i = 0
            ep_len = len(acs_1)
            for transition in range(ep_len-1):
                total_buffer_size += 1
                obs_temp = obs_1[transition].get('observation').copy()
                if 'Maze' in self.env_name:
                    obs_temp = np.array(obs_temp[:3].tolist() + maze_array.copy())
                obs.append(np.array([obs_temp.copy()]))
                acts.append(np.array([acs_1[transition].copy()]))
                rews.append(np.array([0]))#rew_1[env_index][transition]])
                goals.append(np.array([obs_1[transition].get('desired_goal').copy()]))
                achieved_goals.append(np.array([obs_1[transition].get('achieved_goal')].copy()))
                env_indexes.append(np.array([np.array([env_index])].copy()))
                rew = []

            obs_temp = obs_1[ep_len-1].get('observation').copy()
            if 'Maze' in self.env_name:
                obs_temp = np.array(obs_temp[:3].tolist() + maze_array.copy())
            obs.append([obs_temp.copy()])
            achieved_goals.append([obs_1[ep_len-1].get('achieved_goal').copy()])

            episode = dict(o=obs,
                           u=acts,
                           g=goals,
                           r=rews,
                           ag=achieved_goals,
                            env_indexes=env_indexes)

            episode = convert_episode_to_batch_major(episode)
            predictBuffer.store_episode(episode)

            if update_stats:
                # add transitions to normalizer to normalize the demo data as well
                episode['o_2'] = episode['o'][:, 1:, :]
                episode['ag_2'] = episode['ag'][:, 1:, :]
                num_normalizing_transitions = transitions_in_episode_batch(episode)
                transitions = self.sample_transitions(episode, num_normalizing_transitions)

                o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
                transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)

                self.o_stats.update(transitions['o'])
                self.g_stats.update(transitions['g'])

                self.o_stats.recompute_stats()
                self.g_stats.recompute_stats()
            episode.clear()
        del obs_1_list
        del acs_1_list
        print("Predict level demo buffer size: ", total_buffer_size)


    def initDemoBufferUpper(self, demoDataFile, update_stats=True):
        # Initialize upper level expert demonstration buffer
        print('Upper level data loading from', demoDataFile)
        demoData = np.load(demoDataFile, allow_pickle=True)
        info_keys = [key.replace('info_', '') for key in self.input_dims.keys() if key.startswith('info_')]
        info_values = [np.empty((self.T, self.rollout_batch_size, self.input_dims['info_' + key]), np.float32) for key in info_keys]
        self.num_demo = demoData['obs'].shape[0]
        obs_1 = demoData['obs']
        acs_1 = demoData['acs']
        rew_1 = demoData['rew']
        info_1 = demoData['info']

        for epsd in tqdm(range(self.num_demo)):
            obs, acts, goals, achieved_goals , rews = [], [] ,[] ,[], []
            i = 0
            ep_len = len(acs_1[epsd])
            for transition in range(ep_len-1):
                obs.append([obs_1[epsd ][transition].get('observation')])
                acts.append([acs_1[epsd][transition]])
                rews.append([rew_1[epsd][transition]])
                goals.append([obs_1[epsd][transition].get('desired_goal')])
                achieved_goals.append([obs_1[epsd][transition].get('achieved_goal')])
                rew = []
                for idx, key in enumerate(info_keys):
                    if key != 'env_index':
                        info_values[idx][transition, i] = info_1[epsd][transition][key]

            obs.append([obs_1[epsd][ep_len-1].get('observation')])
            achieved_goals.append([obs_1[epsd][ep_len-1].get('achieved_goal')])
            if np.array(acts).shape[0] == 0:
                continue
            episode = dict(o=obs,
                           u=acts,
                           g=goals,
                           r=rews,
                           ag=achieved_goals)
            for key, value in zip(info_keys, info_values):
                episode['info_{}'.format(key)] = value

            episode = convert_episode_to_batch_major(episode)
            demoBuffer1.store_episode(episode)

            if update_stats:
                # add transitions to normalizer to normalize the demo data as well
                episode['o_2'] = episode['o'][:, 1:, :]
                episode['ag_2'] = episode['ag'][:, 1:, :]
                num_normalizing_transitions = transitions_in_episode_batch(episode)
                transitions = self.sample_transitions(episode, num_normalizing_transitions)

                o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
                o_list = []
                p_list = []
                if self.is_image_data:
                    for i in range(len(o)):
                        o_list.append(self.get_image_obs(o[i], g[i]))
                        p_list.append(o[i])
                else:
                    for i in range(len(o)):
                        o_list.append(o[i])
                o_list = np.array(o_list)
                o = o_list
                if self.is_image_data:
                    p_list = np.array(p_list)
                    p = p_list
                    temp1, temp2 = self._preprocess_og(p, ag, g)
                transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
                # No need to preprocess the o_2 and g_2 since this is only used for stats
                self.o_stats.update(transitions['o'])
                self.g_stats.update(transitions['g'])

                self.o_stats.recompute_stats()
                self.g_stats.recompute_stats()
                if self.is_image_data:
                    self.p_stats.update(temp1)
                    self.p_stats.recompute_stats()
            episode.clear()
        print("Upper level demo buffer size: ", demoBuffer1.get_current_size())


    def initDemoBufferUpperMultiple(self, demoDataFile, update_stats=True, limit_data=-1):
        # Initialize upper level expert demonstration buffer
        print('Upper level data loading from', demoDataFile)
        max_drag_angle = 0
        min_drag_angle = 0
        self.update_stats_counter = 0
        demoData = np.load(demoDataFile, allow_pickle=True)
        total_buffer_size = 0
        info_keys = [key.replace('info_', '') for key in self.input_dims.keys() if key.startswith('info_')]
        info_values = [np.empty((self.T, self.rollout_batch_size, self.input_dims['info_' + key]), np.float32) for key in info_keys]
        obs_1_list = demoData['obs']
        acs_1_list = demoData['acs']
        # del demoData

        if 'Rope' in self.env_name:
            max_u = np.array([])
            action_offset = np.array([])
            for i in range(15):
                max_u = np.append(max_u, [0.25, 0.27], axis=0)
            for i in range(15):
                action_offset = np.append(action_offset, [1.3, 0.75], axis=0)
        elif 'kitchen' in self.env_name:
            max_u = [1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,0.88,0.01,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,0.75,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6]
            action_offset = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        else:
            max_u = [0.25, 0.27, 0.145]
            action_offset = [1.3, 0.75, 0.555]

        num_env = self.num_upper_demos
        # if limit_data == -1:
        #     num_env = len(acs_1_list)
        # else:
        #     num_env = limit_data
        for env_index in tqdm(range(num_env)):
            if 'Maze' in self.env_name:
                self.env.env.setIndex(env_index)
                maze_array = self.env.env.get_maze_array().ravel().tolist().copy()
                gates = self.env.env.gates.copy()
            num_demo = len(obs_1_list[env_index])
            obs_1 = obs_1_list[env_index].copy()
            acs_1 = acs_1_list[env_index].copy()

            for epsd in range(num_demo):
                obs, acts, goals, achieved_goals , rews, env_indexes = [], [] ,[] ,[], [], []
                i = 0
                ep_len = len(acs_1[epsd])
                for transition in range(ep_len):
                    total_buffer_size += 1
                    obs_temp = obs_1[epsd ][transition].get('observation').copy()
                    goal_temp = obs_1[epsd ][transition].get('desired_goal').copy()
                    if 'Maze' in self.env_name:
                        obs_temp = np.array(obs_temp[:3].tolist() + maze_array.copy()) #+ maze_array.copy()
                    obs.append(np.array([obs_temp.copy()]))
                    if 'Rope' in self.env_name:
                        action_copy_temp = acs_1[epsd][transition].copy()
                        action_copy_temp = (action_offset + (max_u * action_copy_temp)).copy()
                        act_copy = self.rope_obs_to_action_encoder(action_copy_temp).copy()
                    else:
                        act_copy = acs_1[epsd][transition].copy()
                    acts.append(np.array([act_copy]))
                    rews.append(np.array([0]))#rew_1[epsd][transition]])
                    goals.append(np.array([obs_1[epsd][transition].get('desired_goal').copy()]))
                    achieved_goals.append(np.array([obs_1[epsd][transition].get('achieved_goal')].copy()))
                    env_indexes.append(np.array([np.array([env_index])].copy()))
                    rew = []

                total_buffer_size += 1
                obs_temp = obs_1[epsd ][-1].get('observation').copy()
                goal_temp = obs_1[epsd ][-1].get('desired_goal').copy()
                if 'Maze' in self.env_name:
                    obs_temp = np.array(obs_temp[:3].tolist() + maze_array.copy()) #+ maze_array.copy()
                obs.append([obs_temp.copy()])
                achieved_goals.append([obs_1[epsd][-1].get('achieved_goal').copy()])
                if np.array(acts).shape[0] == 0:
                    continue
                episode = dict(o=obs,
                               u=acts,
                               g=goals,
                               r=rews,
                               ag=achieved_goals,
                               env_indexes=env_indexes)
                for key, value in zip(info_keys, info_values):
                    episode['info_{}'.format(key)] = value

                episode = convert_episode_to_batch_major(episode)
                demoBuffer1.store_episode(episode)

                if update_stats and self.update_stats_counter < 500:
                    self.update_stats_counter += 1
                    # add transitions to normalizer to normalize the demo data as well
                    episode['o_2'] = episode['o'][:, 1:, :]
                    # print(episode['o_2'].shape)
                    episode['ag_2'] = episode['ag'][:, 1:, :]
                    num_normalizing_transitions = transitions_in_episode_batch(episode)
                    transitions = self.sample_transitions(episode, num_normalizing_transitions)

                    o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
                    o_list = []
                    p_list = []
                    if self.is_image_data:
                        for i in range(len(o)):
                            o_list.append(o[i][3:])
                            p_list.append(o[i][:3])
                    else:
                        for i in range(len(o)):
                            o_list.append(o[i])
                    o_list = np.array(o_list)
                    o = o_list
                    if self.is_image_data:
                        p_list = np.array(p_list)
                        p = p_list
                        temp1, temp2 = self._preprocess_og(p, ag, g)
                    transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
                    
                    # No need to preprocess the o_2 and g_2 since this is only used for stats

                    self.o_stats.update(transitions['o'])
                    self.g_stats.update(transitions['g'])

                    self.o_stats.recompute_stats()
                    self.g_stats.recompute_stats()
                    if self.is_image_data:
                        self.p_stats.update(temp1)
                        self.p_stats.recompute_stats()
                episode.clear()
        del obs_1_list
        del acs_1_list
        print("Upper level demo buffer loaded. Size: ", total_buffer_size)

    def initDemoBufferUpperMultipleCurriculum(self, obss, acss, update_stats=True):
        # Initialize upper level expert demonstration buffer (curriculum)
        import time
        start = time.time()
        self.update_stats_counter = 0
        max_drag_angle = 0
        min_drag_angle = 0
        total_buffer_size = 0
        info_keys = [key.replace('info_', '') for key in self.input_dims.keys() if key.startswith('info_')]
        info_values = [np.empty((self.T, self.rollout_batch_size, self.input_dims['info_' + key]), np.float32) for key in info_keys]
        obs_1_list = obss
        acs_1_list = acss
        if 'Rope' in self.env_name:
            max_u = np.array([])
            action_offset = np.array([])
            for i in range(15):
                max_u = np.append(max_u, [0.25, 0.27], axis=0)
            for i in range(15):
                action_offset = np.append(action_offset, [1.3, 0.75], axis=0)
        elif 'kitchen' in self.env_name:
            max_u = [1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,0.88,0.01,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,0.75,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6]
            action_offset = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        else:
            max_u = [0.25, 0.27, 0.145]
            action_offset = [1.3, 0.75, 0.555]
        num_env = self.num_upper_demos
        num_env = len(acs_1_list)
        for env_index in tqdm(range(num_env)):
            if 'Maze' in self.env_name:
                self.env.env.setIndex(env_index)
                maze_array = self.env.env.get_maze_array().ravel().tolist().copy()
                gates = self.env.env.gates.copy()
            num_demo = len(obs_1_list[env_index])
            obs_1 = obs_1_list[env_index].copy()
            acs_1 = acs_1_list[env_index].copy()
            for epsd in range(num_demo):
                obs, acts, goals, achieved_goals , rews, env_indexes = [], [] ,[] ,[], [], []
                i = 0
                ep_len = len(acs_1[epsd])
                for transition in range(ep_len):
                    total_buffer_size += 1
                    obs_temp = obs_1[epsd ][transition].get('observation').copy()
                    goal_temp = obs_1[epsd ][transition].get('desired_goal').copy()
                    if 'Maze' in self.env_name:
                        obs_temp = np.array(obs_temp[:3].tolist() + maze_array.copy()) #+ maze_array.copy()
                    obs.append(np.array([obs_temp.copy()]))
                    if 'Rope' in self.env_name:
                        action_copy_temp = acs_1[epsd][transition].copy()
                        action_copy_temp = (action_offset + (max_u * action_copy_temp)).copy()
                        act_copy = self.rope_obs_to_action_encoder(action_copy_temp).copy()
                    else:
                        act_copy = acs_1[epsd][transition].copy()
                    acts.append(np.array([act_copy]))
                    rews.append(np.array([0]))#rew_1[epsd][transition]])
                    goals.append(np.array([obs_1[epsd][transition].get('desired_goal').copy()]))
                    achieved_goals.append(np.array([obs_1[epsd][transition].get('achieved_goal')].copy()))
                    env_indexes.append(np.array([np.array([env_index])].copy()))
                    rew = []
                total_buffer_size += 1
                obs_temp = obs_1[epsd ][-1].get('observation').copy()
                goal_temp = obs_1[epsd ][-1].get('desired_goal').copy()
                if 'Maze' in self.env_name:
                    obs_temp = np.array(obs_temp[:3].tolist() + maze_array.copy()) #+ maze_array.copy()
                obs.append([obs_temp.copy()])
                achieved_goals.append([obs_1[epsd][-1].get('achieved_goal').copy()])
                if np.array(acts).shape[0] == 0:
                    continue
                episode = dict(o=obs,
                               u=acts,
                               g=goals,
                               r=rews,
                               ag=achieved_goals,
                               env_indexes=env_indexes)
                for key, value in zip(info_keys, info_values):
                    episode['info_{}'.format(key)] = value
                episode = convert_episode_to_batch_major(episode)
                demoBuffer1.store_episode(episode)
                if update_stats and self.update_stats_counter < 500:
                    self.update_stats_counter += 1
                    # add transitions to normalizer to normalize the demo data as well
                    episode['o_2'] = episode['o'][:, 1:, :]
                    episode['ag_2'] = episode['ag'][:, 1:, :]
                    num_normalizing_transitions = transitions_in_episode_batch(episode)
                    transitions = self.sample_transitions(episode, num_normalizing_transitions)
                    o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
                    o_list = []
                    p_list = []
                    if self.is_image_data:
                        for i in range(len(o)):
                            o_list.append(o[i][3:])
                            p_list.append(o[i][:3])
                    else:
                        for i in range(len(o)):
                            o_list.append(o[i])
                    o_list = np.array(o_list)
                    o = o_list
                    if self.is_image_data:
                        p_list = np.array(p_list)
                        p = p_list
                        temp1, temp2 = self._preprocess_og(p, ag, g)
                    transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
                    
                    # No need to preprocess the o_2 and g_2 since this is only used for stats
                    self.o_stats.update(transitions['o'])
                    self.g_stats.update(transitions['g'])
                    self.o_stats.recompute_stats()
                    self.g_stats.recompute_stats()
                    if self.is_image_data:
                        self.p_stats.update(temp1)
                        self.p_stats.recompute_stats()
                episode.clear()
        del obs_1_list
        del acs_1_list
        logger.info("Populating. Upper level demo buffer loaded. Size: ", total_buffer_size)

    def rope_action_to_obs_encoder(self, u_temp):
        # Rope action to observation encoder (ideally should be part of rope environment, added here for easy calling)
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

    def rope_obs_to_action_encoder(self, obs_temp):
        # Rope observation to action encoder (ideally should be part of rope environment, added here for easy calling)
        max_angle = 45.
        act_copy = [0,0]
        for i in range(1,14):
            x1, y1 = obs_temp[2*(i-1)], obs_temp[2*(i-1)+1]
            x2, y2 = obs_temp[2*(i)], obs_temp[2*(i)+1]
            slope1 = (y2-y1) / (x2-x1)
            angle_radian1 = np.arctan(slope1)
            angle_degrees1 = np.degrees(angle_radian1)
            drag_angle1 = angle_degrees1

            x1, y1 = obs_temp[2*(i)], obs_temp[2*(i)+1]
            x2, y2 = obs_temp[2*(i+1)], obs_temp[2*(i+1)+1]
            slope2 = (y2-y1) / (x2-x1)
            angle_radian2 = np.arctan(slope2)
            angle_degrees2 = np.degrees(angle_radian2)
            drag_angle2 = angle_degrees2

            drag_angle = drag_angle2 - drag_angle1

            if np.absolute(drag_angle) > 90:
                drag_angle = 180 + drag_angle

            if drag_angle > 90:
                drag_angle = drag_angle - 360
            if drag_angle < -90:
                drag_angle = drag_angle + 360
            act_copy.append(drag_angle)

        act_copy = np.array(act_copy)/max_angle
        return act_copy.copy()

    def add_upper_demo_buffer(self, demoDataLower, update_stats=True):
        # Legacy script for adding demonstration to buffer
        info_keys = [key.replace('info_', '') for key in self.input_dims.keys() if key.startswith('info_')]
        info_values = [np.empty((self.T, self.rollout_batch_size, self.input_dims['info_' + key]), np.float32) for key in info_keys]
        
        policy = self.otherPolicy
        env = self.env

        if 'Rope' in self.env_name:
            max_u = np.array([])
            action_offset = np.array([])
            for i in range(15):
                max_u = np.append(max_u, [0.24, 0.27, 0.145], axis=0)
            for i in range(15):
                action_offset = np.append(action_offset, [1.29, 0.75, 0.555], axis=0)
        elif 'kitchen' in self.env_name:
            max_u = [1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,0.88,0.01,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,0.75,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6]
            action_offset = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        else:
            max_u = self.max_u
            action_offset = self.action_offset
        actions = []
        observations = []
        rewards = []
        infos = []

        episodeAcs = []
        episodeRew = []
        episodeObs = []
        episodeInfo = []

        transition = -1
        num_demo = len(demoDataLower['acs'])
        ep_len = len(demoDataLower['acs'][1])
        epsd = np.random.randint(num_demo)
        demo_episodes_obs = demoDataLower['obs'][epsd]
        goal = demo_episodes_obs[0].get('desired_goal')
        transition = 0
        while transition < ep_len-1:
            initial_obs = demo_episodes_obs[transition]
            episodeObs.append(initial_obs)
            block_reward = 0
            curr_obs = initial_obs.get('observation').reshape(1,-1)
            curr_ag = initial_obs.get('achieved_goal').reshape(1,-1)
            timestep_len = 0
            while block_reward == 0 and transition < ep_len-1 and timestep_len < 100:
                timestep_len += 1
                env.env._set_state(initial_obs.get('observation'), goal)
                transition += 1
                goal_lower = demo_episodes_obs[transition].get('achieved_goal').reshape(1,-1)
                for i in range(4):
                    policy_output = policy.get_actions(curr_obs, curr_ag, goal_lower)
                    if policy_output.ndim == 1:
                        policy_output = policy_output.reshape(1, -1)
                    env_obs, _, _, info = env.step(policy_output[0])
                    curr_obs = env_obs.get('observation').reshape(1,-1)
                    curr_ag = env_obs.get('achieved_goal').reshape(1,-1)
                    block_reward = self.compute_reward(curr_ag, goal_lower)
                    if block_reward == 0:
                        break
                goal_reward = self.compute_reward(curr_ag, goal.reshape(1,-1))
                if goal_reward == 0:
                    break

            episodeInfo.append(info)
            action_temp = (goal_lower[0] - action_offset) / [x for x in max_u]
            action = np.zeros(4)
            action[:3] = action_temp
            episodeAcs.append(action)
            episodeRew.append(block_reward)

            goal_reward = self.compute_reward(curr_ag, goal.reshape(1,-1))
            if goal_reward == 0:
                break

        # Add last goal
        episodeObs.append(initial_obs)
        episodeInfo.append(info)
        goal_lower = demo_episodes_obs[ep_len-1].get('achieved_goal')
        action_temp = (goal_lower - action_offset) / [x for x in max_u]
        action = np.zeros(4)
        action[:3] = action_temp
        episodeAcs.append(action)
        episodeRew.append(block_reward)

        # print("Episode len",len(episodeObs))
        actions.append(episodeAcs)
        rewards.append(episodeRew)
        observations.append(episodeObs)
        infos.append(episodeInfo)

        demoData = dict(acs=actions, obs=observations, info=infos, rew=rewards)

        obs, acts, goals, achieved_goals , rews = [], [] ,[] ,[], []
        i = 0
        ep_len = len(demoData['acs'][0])-1
        obs_1 = demoData['obs']
        acs_1 = demoData['acs']
        rew_1 = demoData['rew']
        info_1 = demoData['info']

        for transition in range(ep_len):
            obs.append([obs_1[0][transition].get('observation')])
            acts.append([acs_1[0][transition]])
            rews.append([rew_1[0][transition]])
            goals.append([obs_1[0][transition].get('desired_goal')])
            achieved_goals.append([obs_1[0][transition].get('achieved_goal')])
            rew = []
            for idx, key in enumerate(info_keys):
                info_values[idx][transition, i] = info_1[0][transition][key]

        obs.append([obs_1[0][ep_len].get('observation')])
        achieved_goals.append([obs_1[0][ep_len].get('achieved_goal')])

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       r=rews,
                       ag=achieved_goals)
        for key, value in zip(info_keys, info_values):
            episode['info_{}'.format(key)] = value

        episode = convert_episode_to_batch_major(episode)
        self.store_episode_upper(episode)
        self.store_episode(episode)


    def compute_reward(self, achieved_goal, goal):
        '''
        Compute reward goal and the achieved goal.
        Inputs: achieved goal and final goal
        Output: rewards
        '''
        if 'Rope' in self.env_name:
            distance_threshold = 0.1
        else:
            distance_threshold = 0.05
        d = self.goal_distance(achieved_goal, goal)
        return -(d > distance_threshold).astype(np.float32)

    def goal_distance(self, goal_a, goal_b):
        '''
        Compute goal distance
        Inputs: goal a and goal b
        Output: goal distance between input goals
        '''
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def get_current_min_dist(self):
        return self.min_dist

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """
        self.buffer.store_episode(episode_batch)
        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions, self.optimal_policy, self.lower_policy)

            o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
            o_list = []
            p_list = []
            if self.is_image_data:
                for i in range(len(o)):
                    o_list.append(o[i][3:])
                    p_list.append(o[i][:3])
            else:
                for i in range(len(o)):
                    o_list.append(o[i])
            o_list = np.array(o_list)
            o = o_list
            if self.is_image_data:
                p_list = np.array(p_list)
                p = p_list
                temp1, temp2 = self._preprocess_og(p, ag, g)
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()
            if self.is_image_data:
                # print(temp1)
                self.p_stats.update(temp1)
                self.p_stats.recompute_stats()

    def store_episode_predict(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """
        predictBuffer.store_episode(episode_batch)
        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

            o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
            o_list = []
            p_list = []
            if self.is_image_data:
                for i in range(len(o)):
                    o_list.append(o[i][3:])
                    p_list.append(o[i][:3])
            else:
                for i in range(len(o)):
                    o_list.append(o[i])
            o_list = np.array(o_list)
            o = o_list
            if self.is_image_data:
                p_list = np.array(p_list)
                p = p_list
                temp1, temp2 = self._preprocess_og(p, ag, g)
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()
            if self.is_image_data:
                # print(temp1)
                self.p_stats.update(temp1)
                self.p_stats.recompute_stats()

    def store_episode_upper(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """
        demoBuffer1.store_episode(episode_batch)
        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

            o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
            o_list = []
            p_list = []
            if self.is_image_data:
                for i in range(len(o)):
                    o_list.append(o[i][3:])
                    p_list.append(o[i][:3])
            else:
                for i in range(len(o)):
                    o_list.append(o[i])
            o_list = np.array(o_list)
            o = o_list
            if self.is_image_data:
                p_list = np.array(p_list)
                p = p_list
                temp1, temp2 = self._preprocess_og(p, ag, g)
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            # No need to preprocess the o_2 and g_2 since this is only used for stats

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()
            if self.is_image_data:
                # print(temp1)
                self.p_stats.update(temp1)
                self.p_stats.recompute_stats()

    def get_image_obs(self, obs, goal, env_index=-1, focus=False):
        focus_width = 15
        if env_index == -1:
            env_index = [self.env.env.index]
        env_index = env_index[0]
        is_dobot = 'Dobot' in self.env_name
        grip_pos = obs[:3].copy()
        maze_array = obs[3:124].copy()
        maze_array = maze_array.reshape(11,11)
        goal_pos = goal
        blackBox = grip_pos.copy()
        redCircle = goal_pos.copy()
        height = 30
        width = 30
        imgGripper = np.zeros((height,width), np.uint8)
        imgWall = np.zeros((height,width), np.uint8)
        imgGoal = np.zeros((height,width), np.uint8)
        half_block_len = 1.2
        gripper_len = 1.2
        sphere_rad = 1.2
        width, height = maze_array.shape
        for i in range(width):
            for j in range(height):
                # flag += 1
                if maze_array[i,j]:
                    imgWall[3*i:3*(i+1),int(2.76*(j)):int(2.76*(j+1))] = 255 

        image = imgWall
        if focus:
            min_x = min(max(0, xx_gripper-focus_width), width-2*focus_width)
            min_y = min(max(0, yy_gripper-focus_width), height-2*focus_width)
            max_x = min(min_x + 2*focus_width, width)
            max_y = min(min_y + 2*focus_width, height)
            imgWall = imgWall[min_x:max_x, min_y:max_y]
            image = imgWall.reshape(2*focus_width, 2*focus_width, 1)
            
        image = image/255.0
        image = image.astype(np.uint8)
        obs = image.ravel().copy()
        return obs

    def map_cor(self, pos, X1=0.0,X2=30.0,Y1=0.,Y2=30.0,x1=1.05,x2=1.55,y1=0.48,y2=1.02):
        x = pos[0]
        y = pos[1]

        X = X1 + ( (x-x1) * (X2-X1) / (x2-x1) )
        Y = Y1 + ( (y-y1) * (Y2-Y1) / (y2-y1) )
        return(np.array([X,Y]))

    def map_cor_dobot(self, pos, X1=0.0,X2=50.0,Y1=0.,Y2=50.0,x2=0.575,x1=0.795,y1=0.6,y2=1):
        # Dobot
        x = pos[1]
        y = pos[0]
        if x<0.57 or x>0.8 or y<0.55 or y>1.05:
            return np.array([290,480])

        X = X1 + ( (x-x1) * (X2-X1) / (x2-x1) )
        Y = Y1 + ( (y-y1) * (Y2-Y1) / (y2-y1) )
        return(np.array([X,Y]))

    def store_hrl_good_transition(self, obs0, obs2, action, reward, goal):
        # Save good transition in memory buffer
        self.hrl_buffer.append(obs0, obs2, action, reward, goal)

    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()
        if (self.scope != 'sac0' and self.bc_loss_upper) or (self.scope == 'sac0' and self.bc_loss):
            if self.adversarial_loss:
                self.discriminator_adam.sync()
        if self.scope != 'sac0' and self.predictor_loss:
            self.predictor_adam.sync()
        if self.reward_model:
            self.reward_adam.sync()

    def _grads_adversarial_predict(self):
        # Avoid feed_dict here for performance!
        critic_loss , actor_loss, discriminator_loss, predictor_loss, Q_grad, pi_grad, discriminator_grad, predictor_grad = self.sess.run([
            self.Q_loss_tf,
            self.pi_loss_tf,
            self.discriminator_loss_tf,
            self.predictor_loss_tf,
            self.Q_grad_tf,
            self.pi_grad_tf,
            self.discriminator_grad_tf,
            self.predictor_grad_tf,
        ])
        return critic_loss , actor_loss, discriminator_loss, predictor_loss, Q_grad, pi_grad, discriminator_grad, predictor_grad

    def _grads_adversarial(self):
        # Avoid feed_dict here for performance!
        critic_loss , actor_loss, discriminator_loss, Q_grad, pi_grad, discriminator_grad= self.sess.run([
            self.Q_loss_tf,
            self.pi_loss_tf,
            self.discriminator_loss_tf,
            self.Q_grad_tf,
            self.pi_grad_tf,
            self.discriminator_grad_tf,
        ])
        return critic_loss , actor_loss, discriminator_loss, Q_grad, pi_grad, discriminator_grad

    def _grads_predict(self):
        # Avoid feed_dict here for performance!
        critic_loss , actor_loss, predictor_loss, Q_grad, pi_grad, predictor_grad = self.sess.run([
            self.Q_loss_tf,
            self.pi_loss_tf,
            self.predictor_loss_tf,
            self.Q_grad_tf,
            self.pi_grad_tf,
            self.predictor_grad_tf,
        ])
        return critic_loss , actor_loss, predictor_loss, Q_grad, pi_grad, predictor_grad

    def _grads_reward(self):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, reward_loss, Q_grad, pi_grad, reward_grad = self.sess.run([
            self.Q_loss_tf,
            self.pi_loss_tf,
            self.reward_loss_tf,
            self.Q_grad_tf,
            self.pi_grad_tf,
            self.reward_grad_tf,
        ])
        return critic_loss, actor_loss, reward_loss, Q_grad, pi_grad, reward_grad

    def _grads(self):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run([
            self.Q_loss_tf,
            self.pi_loss_tf,
            self.Q_grad_tf,
            self.pi_grad_tf,
        ])
        return critic_loss, actor_loss, Q_grad, pi_grad

    def access_global_demo_buffer(self):
        temp_buffer = None
        if self.scope == 'sac0':
            global demoBuffer0
            temp_buffer = demoBuffer0
        else:
            global demoBuffer1
            temp_buffer = demoBuffer1
        return temp_buffer

    def _update_adversarial_predict(self, Q_grad, pi_grad, discriminator_grad, predictor_grad=None):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)
        self.discriminator_adam.update(discriminator_grad, self.pi_lr)
        self.predictor_adam.update(predictor_grad, self.pi_lr)

    def _update_adversarial(self, Q_grad, pi_grad, discriminator_grad, predictor_grad=None):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)
        self.discriminator_adam.update(discriminator_grad, self.pi_lr)

    def _update_predict(self, Q_grad, pi_grad, discriminator_grad, predictor_grad=None):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)
        self.predictor_adam.update(predictor_grad, self.pi_lr)

    def _update_reward(self, Q_grad, pi_grad, reward_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)
        self.reward_adam.update(reward_grad, self.pi_lr)

    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    def sample_batch(self):
        # Sample a batch for forward pass
        if self.scope != 'sac0':
            if self.bc_loss_upper:
                if self.predictor_loss and self.adversarial_loss:
                    transitions = self.buffer.sample(self.batch_size - 2 * self.adversarial_predict_batch_size)
                    transitionsDiscriminatorDemo = demoBuffer1.sample(self.adversarial_predict_batch_size)
                    for k, values in transitionsDiscriminatorDemo.items():
                        transitions[k] = np.append(transitions[k], values, axis=0)
                    transitionsPredictDemo = predictBuffer.sample(self.adversarial_predict_batch_size)
                    for k, values in transitionsPredictDemo.items():
                        transitions[k] = np.append(transitions[k], values, axis=0)
                elif self.predictor_loss:
                    transitions = self.buffer.sample(self.batch_size - self.self.predict_batch_size)
                    transitionspredictDemo = predictBuffer.sample(self.predict_batch_size)
                    for k, values in transitionspredictDemo.items():
                        transitions[k] = np.append(transitions[k], values, axis=0)
                else:
                    if self.adversarial_loss:
                        transitions = self.buffer.sample(self.batch_size - self.adversarial_batch_size)
                        transitionsDemo = demoBuffer1.sample(self.adversarial_batch_size)
                        for k, values in transitionsDemo.items():
                            transitions[k] = np.append(transitions[k], values, axis=0)
                    else:
                        if not self.upper_only_bc:
                            transitions = self.buffer.sample(self.batch_size - self.demo_batch_size)
                            transitionsDemo = demoBuffer1.sample(self.demo_batch_size)
                            for k, values in transitionsDemo.items():
                                transitions[k] = np.append(transitions[k], values, axis=0)
                        else:
                            transitions = demoBuffer1.sample(self.demo_batch_size)
            else:
                transitions = self.buffer.sample(self.batch_size)
        else:
            if self.bc_loss:
                if self.adversarial_loss:
                    transitions = self.buffer.sample(self.batch_size - self.adversarial_batch_size)
                    transitionsDemo = demoBuffer0.sample(self.adversarial_batch_size)
                    for k, values in transitionsDemo.items():
                        transitions[k] = np.append(transitions[k], values, axis=0)
                else:
                    if not self.upper_only_bc:
                        transitions = self.buffer.sample(self.batch_size - self.demo_batch_size)
                        transitionsDemo = demoBuffer0.sample(self.demo_batch_size)
                        for k, values in transitionsDemo.items():
                            transitions[k] = np.append(transitions[k], values, axis=0)
                    else:
                        transitions = demoBuffer0.sample(self.demo_batch_size)
            else:
                transitions = self.buffer.sample(self.batch_size)

        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        if self.reward_model:
            reward_o_1 = transitions['reward_o_1']
            reward_o_2 = transitions['reward_o_2']
            reward_g_1 = transitions['reward_g_1']
            reward_g_2 = transitions['reward_g_2']

        o_list = []
        p_list = []
        for i in range(len(o)):
            if self.is_image_data:
                o_pixel = o[i][3:].copy()
                p_pixel = o[i][:3].copy()
                p_list.append(p_pixel)
            else:
                o_pixel = o[i].copy()
            o_list.append(o_pixel)
        o_list = np.array(o_list)
        o = o_list
        if self.is_image_data:
            p_list = np.array(p_list)
            p = p_list

        o_2_list = []
        p_2_list = []
        for i in range(len(o_2)):
            if self.is_image_data:
                o_pixel_2 = o_2[i][3:].copy()
                p_pixel_2 = o_2[i][:3].copy()
                p_2_list.append(p_pixel_2)
            else:
                o_pixel_2 = o_2[i].copy()
            o_2_list.append(o_pixel_2)
        o_2_list = np.array(o_2_list)
        o_2 = o_2_list
        if self.is_image_data:
            p_2_list = np.array(p_2_list)
            p_2 = p_2_list
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)
        if self.is_image_data:
            transitions['p'], transitions['g'] = self._preprocess_og(p, ag, g)
            transitions['p_2'], transitions['g_2'] = self._preprocess_og(p_2, ag_2, g)

        if self.reward_model:
            transitions['reward_o_1'] = self._preprocess_og_no_ag_reward(reward_o_1)
            transitions['reward_o_2'] = self._preprocess_og_no_ag_reward(reward_o_2)
            transitions['reward_g_1'] = self._preprocess_og_no_ag_reward(reward_g_1)
            transitions['reward_g_2'] = self._preprocess_og_no_ag_reward(reward_g_2)

        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]

        return transitions_batch

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
            assert len(self.buffer_ph_tf) == len(batch)
            self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def train(self, stage=True):
        if stage:
            self.stage_batch()
        if self.scope != 'sac0':
            if self.bc_loss_upper and self.adversarial_loss and self.predictor_loss:
                critic_loss, actor_loss, discriminator_loss, predictor_loss, Q_grad, pi_grad, discriminator_grad, predictor_grad = self._grads_adversarial_predict()
                self._update_adversarial_predict(Q_grad, pi_grad, discriminator_grad, predictor_grad)
                return critic_loss, actor_loss, discriminator_loss, predictor_loss
            elif self.bc_loss_upper and self.predictor_loss: 
                critic_loss, actor_loss, predictor_loss, Q_grad, pi_grad, predictor_grad = self._grads_predict()
                self._update_predict(Q_grad, pi_grad, predictor_grad)
                return critic_loss, actor_loss, predictor_loss
            elif self.bc_loss_upper and self.adversarial_loss:
                critic_loss, actor_loss, discriminator_loss, Q_grad, pi_grad, discriminator_grad = self._grads_adversarial()
                self._update_adversarial(Q_grad, pi_grad, discriminator_grad)
                return critic_loss, actor_loss, discriminator_loss
            else:
                if self.reward_model:
                    critic_loss, actor_loss, reward_loss, Q_grad, pi_grad, reward_grad = self._grads_reward()
                    self._update_reward(Q_grad, pi_grad, reward_grad)
                    return critic_loss, actor_loss, reward_loss
                else:
                    critic_loss, actor_loss, Q_grad, pi_grad = self._grads()
                    self._update(Q_grad, pi_grad)
                    return critic_loss, actor_loss
        else:
            if self.bc_loss and self.adversarial_loss:
                critic_loss, actor_loss, discriminator_loss, Q_grad, pi_grad, discriminator_grad = self._grads_adversarial()
                self._update_adversarial(Q_grad, pi_grad, discriminator_grad)
                return critic_loss, actor_loss, discriminator_loss
            else:
                critic_loss, actor_loss, Q_grad, pi_grad = self._grads()
                self._update(Q_grad, pi_grad)
                return critic_loss, actor_loss

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def clear_buffer_upper(self):
        global demoBuffer1
        demoBuffer1.clear_buffer()

    def clear_buffer_predict(self):
        global predictBuffer
        predictBuffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    # DQN Nature 2015 paper
    def featuresDQN15(self, hiddens, penulti_linear=512, feature_size=50, reuse=False, scope_name=""):
        dim_image = 10
        model = tf.keras.Sequential()
        model.add(keras.layers.Flatten())
        for hidden in hiddens:
            model.add(keras.layers.Dense(hidden, activation='relu'))
        model.add(keras.layers.Dense(feature_size, activation='tanh'))
        return model

    def get_actions_bc_model(self, obs_input):
        return self.bc_model.predict(obs_input)

    def _create_network(self, reuse=False):
        self.sess = tf.compat.v1.get_default_session()
        if self.sess is None:
            self.sess = tf.compat.v1.InteractiveSession()

        # running averages
        with tf.compat.v1.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            if self.is_image_data:
                self.o_stats = Normalizer(self.image_size, self.norm_eps, self.norm_clip, sess=self.sess)
            else:
                self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.compat.v1.variable_scope('p_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.p_stats = Normalizer(3, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.compat.v1.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        if self.is_image_data:
            batch_tf = OrderedDict([(key, batch[i]) for i, key in enumerate(self.image_shapes.keys())])
        else:
            batch_tf = OrderedDict([(key, batch[i]) for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        if self.scope != 'sac0':
            if self.bc_loss_upper:
                if self.predictor_loss and self.adversarial_loss:
                    mask_adversarial = np.concatenate((np.zeros(self.batch_size - 2*self.adversarial_predict_batch_size), np.ones(self.adversarial_predict_batch_size), np.zeros(self.adversarial_predict_batch_size)), axis = 0)
                    mask_predict = np.concatenate((np.zeros(self.batch_size - self.adversarial_predict_batch_size), np.ones(self.adversarial_predict_batch_size)), axis = 0)
                    mask_adversarial = mask_adversarial.astype(bool)
                    mask_predict = mask_predict.astype(bool)
                    self.discriminator_labels = mask_adversarial
                    self.predictor_labels = mask_predict

                elif self.predictor_loss:
                    mask_predict = np.concatenate((np.zeros(self.batch_size - self.predict_batch_size), np.ones(self.predict_batch_size)), axis = 0)
                    mask_predict = mask_predict.astype(bool)
                    self.predictor_labels = mask_predict

                elif self.adversarial_loss:
                    mask_adversarial = np.concatenate((np.zeros(self.batch_size - self.adversarial_batch_size), np.ones(self.adversarial_batch_size)), axis = 0)
                    mask_adversarial = mask_adversarial.astype(bool)
                    self.discriminator_labels = mask_adversarial
                    if self.margin_loss:
                        mask_pi_adversarial = np.concatenate((np.ones(self.batch_size - self.adversarial_batch_size), np.zeros(self.adversarial_batch_size)), axis = 0)
                        mask_pi_adversarial = mask_pi_adversarial.astype(bool)
                        self.discriminator_pi_labels = mask_pi_adversarial
                else:
                    mask_none = np.concatenate((np.zeros(self.batch_size - self.demo_batch_size), np.ones(self.demo_batch_size)), axis = 0)
                    mask_none = mask_none.astype(bool)
                    self.hrl_labels = mask_none
                    if self.margin_loss:
                        mask_pi_none = np.concatenate((np.ones(self.batch_size - self.demo_batch_size), np.zeros(self.demo_batch_size)), axis = 0)
                        mask_pi_none = mask_pi_none.astype(bool)
                        self.hrl_pi_labels = mask_pi_none
        else:
            if self.bc_loss:
                if self.adversarial_loss:
                    mask_adversarial = np.concatenate((np.zeros(self.batch_size - self.adversarial_batch_size), np.ones(self.adversarial_batch_size)), axis = 0)
                    mask_adversarial = mask_adversarial.astype(bool)
                    self.discriminator_labels = mask_adversarial
                else:
                    mask_none = np.concatenate((np.zeros(self.batch_size - self.demo_batch_size), np.ones(self.demo_batch_size)), axis = 0)
                    mask_none = mask_none.astype(bool)
                    self.hrl_labels = mask_none

        initialize_weights_data = None

        # networks
        with tf.compat.v1.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            if self.is_image_data:
                target_batch_tf['p'] = batch_tf['p_2']
            target_batch_tf['g'] = batch_tf['g_2']
            if self.sac == 0:
                self.main = self.create_actor_critic(batch_tf, initialize_weights_data, net_type='main', **self.__dict__)
            else:
                self.main = self.create_actor_critic(batch_tf, target_batch_tf, initialize_weights_data, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.compat.v1.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            if self.sac == 0:
                self.target = self.create_actor_critic(target_batch_tf, initialize_weights_data, net_type='target', **self.__dict__)
            else:
                target_batch_tf['u'] = self.main.pi_2_tf
                self.target = self.create_actor_critic(target_batch_tf, target_batch_tf, initialize_weights_data, net_type='target', **self.__dict__)
            vs.reuse_variables()

        assert len(self._vars("main")) == len(self._vars("target"))

        # loss functions
        if self.sac == 0:
            target_Q_pi_tf = self.target.Q_pi_tf
            clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
            target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range)
            self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))
        else: 
            #SAC losses
            clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
            # Targets for Q and V regression
            self.min_q_pi_tf = tf.minimum(self.main.Q1_pi_tf,self.main.Q2_pi_tf)# Twin Q networks like in TD3
            self.min_q_targ_tf = tf.minimum(self.target.Q1_tf,self.target.Q2_tf)# Twin Q networks like in TD3
            # Entropy-regularized Bellman backup for Q functions, using Clipped Double-Q targets
            if self.dac:
                discriminator_output = self.main.discriminator_pred_tf
                q_target = tf.stop_gradient(tf.clip_by_value(discriminator_output + self.gamma*(self.min_q_targ_tf - self.alpha * self.main.logp_pi_2_tf), *clip_range))
            else:
                if self.reward_model:
                    q_target = tf.stop_gradient(tf.clip_by_value(self.target.reward_pred_tf + self.gamma*(self.min_q_targ_tf - self.alpha * self.main.logp_pi_2_tf), *clip_range))
                else:
                    q_target = tf.stop_gradient(tf.clip_by_value(batch_tf['r'] + self.gamma*(self.min_q_targ_tf - self.alpha * self.main.logp_pi_2_tf), *clip_range))

            if self.margin_loss and self.scope != 'sac0':
                if self.bc_loss_upper:
                    if self.adversarial_loss:
                        q1_E_loss = 0.5 * tf.reduce_sum(tf.boolean_mask((self.main.Q1_tf), self.discriminator_labels))
                        q2_E_loss = 0.5 * tf.reduce_sum(tf.boolean_mask((self.main.Q2_tf), self.discriminator_labels))
                        q1_pi_loss = 0.5 * tf.reduce_sum(tf.boolean_mask((self.main.Q1_pi_tf), self.discriminator_pi_labels))
                        q2_pi_loss = 0.5 * tf.reduce_sum(tf.boolean_mask((self.main.Q2_pi_tf), self.discriminator_pi_labels))
                    else:
                        q1_E_loss = 0.5 * tf.reduce_sum(tf.boolean_mask((self.main.Q1_tf), self.hrl_labels))
                        q2_E_loss = 0.5 * tf.reduce_sum(tf.boolean_mask((self.main.Q2_tf), self.hrl_labels))
                        q1_pi_loss = 0.5 * tf.reduce_sum(tf.boolean_mask((self.main.Q1_pi_tf), self.hrl_pi_labels))
                        q2_pi_loss = 0.5 * tf.reduce_sum(tf.boolean_mask((self.main.Q2_pi_tf), self.hrl_pi_labels))

            q1_loss = 0.5 * tf.reduce_mean((q_target - self.main.Q1_tf)**2)
            q2_loss = 0.5 * tf.reduce_mean((q_target - self.main.Q2_tf)**2)
            if self.margin_loss and self.scope != 'sac0':
                self.Q_loss_tf = q1_loss + q2_loss + self.margin_lambda*(q1_pi_loss - q1_E_loss) + self.margin_lambda*(q2_pi_loss - q2_E_loss)
            else:
                self.Q_loss_tf = q1_loss + q2_loss

        if self.scope != 'sac0':
            # Upper level
            if self.bc_loss_upper:
                if self.adversarial_loss and self.predictor_loss:
                    self.pi_loss_tf = self.lambda1 * tf.reduce_mean(self.alpha * self.main.logp_pi_tf - self.min_q_pi_tf)
                    self.pi_loss_tf += self.lambda1 * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
                    
                    self.discriminator_loss_tf = tf.reduce_mean(tf.square(2 * self.discriminator_labels - 1 - self.main.discriminator_pred_tf))
                    discriminator_output = self.main.discriminator_pi_pred_tf
                    self.generator_discriminator_loss_tf = tf.reduce_mean(tf.square(discriminator_output - 1))
                    self.pi_loss_tf += self.lambda2 * self.generator_discriminator_loss_tf

                    self.predictor_loss_tf = tf.reduce_mean(tf.square(2 * self.predictor_labels - 1 - self.main.predictor_pred_tf))
                    predictor_output = self.main.predictor_pi_pred_tf
                    self.generator_predictor_loss_tf = tf.reduce_mean(tf.square(predictor_output - 1))
                    self.pi_loss_tf += self.lambda2 * self.generator_predictor_loss_tf

                elif self.predictor_loss:
                    self.pi_loss_tf = self.lambda1 * tf.reduce_mean(self.alpha * self.main.logp_pi_tf - self.min_q_pi_tf)
                    self.pi_loss_tf += self.lambda1 * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
                    
                    self.predictor_loss_tf = tf.reduce_mean(tf.square(2 * self.predictor_labels - 1 - self.main.predictor_pred_tf))
                    predictor_output = self.main.predictor_pi_pred_tf
                    self.generator_predictor_loss_tf = tf.reduce_mean(tf.square(predictor_output - 1))
                    self.pi_loss_tf += self.lambda2 * self.generator_predictor_loss_tf

                elif self.adversarial_loss: 
                    self.pi_loss_tf = self.lambda1 * tf.reduce_mean(self.alpha * self.main.logp_pi_tf - self.min_q_pi_tf)
                    self.pi_loss_tf += self.lambda1 * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
                    
                    self.discriminator_loss_tf = tf.reduce_mean(tf.square(2 * self.discriminator_labels - 1 - self.main.discriminator_pred_tf))
                    discriminator_output = self.main.discriminator_pi_pred_tf
                    self.generator_discriminator_loss_tf = tf.reduce_mean(tf.square(discriminator_output - 1))
                    self.pi_loss_tf += self.lambda2 * self.generator_discriminator_loss_tf

                elif self.q_filter == 1:
                    mask_q_filter = tf.reshape(tf.boolean_mask(self.main.Q1_tf > self.main.Q1_pi_tf, self.hrl_labels), [-1])
                    self.cloning_loss_tf = tf.reduce_sum(tf.square(tf.boolean_mask(tf.boolean_mask((self.main.pi_tf), self.hrl_labels), mask_q_filter)
                             - tf.boolean_mask(tf.boolean_mask((batch_tf['u']), self.hrl_labels), mask_q_filter)))
                    if not self.upper_only_bc:
                        self.pi_loss_tf = self.lambda1 * tf.reduce_mean(self.alpha * self.main.logp_pi_tf - self.min_q_pi_tf)
                        self.pi_loss_tf += self.lambda1 * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
                        self.pi_loss_tf += self.lambda2 * self.cloning_loss_tf
                    else:
                        self.pi_loss_tf = self.lambda2 * self.cloning_loss_tf
                elif self.q_filter == 0:
                    self.cloning_loss_tf = tf.reduce_sum(tf.square(tf.boolean_mask((self.main.pi_tf), self.hrl_labels) - tf.boolean_mask((batch_tf['u']), self.hrl_labels)))
                    self.pi_loss_tf = self.lambda1 * tf.reduce_mean(self.alpha * self.main.logp_pi_tf - self.min_q_pi_tf)
                    self.pi_loss_tf += self.lambda1 * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
                    self.pi_loss_tf += self.lambda2 * self.cloning_loss_tf
            else:
                self.pi_loss_tf = tf.reduce_mean(self.alpha * self.main.logp_pi_tf - self.min_q_pi_tf)
                self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
        else:
            # Lower level
            if self.bc_loss:
                if self.adversarial_loss:
                    self.pi_loss_tf = self.lambda1 * tf.reduce_mean(self.alpha * self.main.logp_pi_tf - self.min_q_pi_tf)
                    self.pi_loss_tf += self.lambda1 * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
                    
                    self.discriminator_loss_tf = tf.reduce_mean(tf.square(2 * self.discriminator_labels - 1 - self.main.discriminator_pred_tf))
                    if not self.dac:
                        discriminator_output = self.main.discriminator_pi_pred_tf
                        self.generator_discriminator_loss_tf = tf.reduce_mean(tf.square(discriminator_output - 1))
                        self.pi_loss_tf += self.lambda2 * self.generator_discriminator_loss_tf

                elif self.q_filter == 1:
                    mask_q_filter = tf.reshape(tf.boolean_mask(self.main.Q1_tf > self.main.Q1_pi_tf, self.hrl_labels), [-1])
                    self.cloning_loss_tf = tf.reduce_sum(tf.square(tf.boolean_mask(tf.boolean_mask((self.main.pi_tf), self.hrl_labels), mask_q_filter)
                             - tf.boolean_mask(tf.boolean_mask((batch_tf['u']), self.hrl_labels), mask_q_filter)))
                    if not self.upper_only_bc:
                        self.pi_loss_tf = self.lambda1 * tf.reduce_mean(self.alpha * self.main.logp_pi_tf - self.min_q_pi_tf)
                        self.pi_loss_tf += self.lambda1 * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
                        self.pi_loss_tf += self.lambda2 * self.cloning_loss_tf
                    else:
                        self.pi_loss_tf = self.lambda2 * self.cloning_loss_tf

                elif self.q_filter == 0:
                    self.cloning_loss_tf = tf.reduce_sum(tf.square(tf.boolean_mask((self.main.pi_tf), self.hrl_labels) - tf.boolean_mask((batch_tf['u']), self.hrl_labels)))
                    self.pi_loss_tf = self.lambda1 * tf.reduce_mean(self.alpha * self.main.logp_pi_tf - self.min_q_pi_tf)
                    self.pi_loss_tf += self.lambda1 * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
                    self.pi_loss_tf += self.lambda2 * self.cloning_loss_tf
            else:
                self.pi_loss_tf = self.lambda1 * tf.reduce_mean(self.alpha * self.main.logp_pi_tf - self.min_q_pi_tf)
                self.pi_loss_tf += self.lambda1 * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))

        if self.reward_model:
            reward_labels_1 = tf.cast(batch_tf['reward_r_1'] > batch_tf['reward_r_2'], tf.float32)
            reward_labels_2 = tf.cast(batch_tf['reward_r_1'] < batch_tf['reward_r_2'], tf.float32)
            reward_equal = tf.cast(tf.equal(reward_labels_1, reward_labels_2), dtype=tf.float32)
            reward_labels_1 = reward_labels_1 + reward_equal*0.5
            reward_labels_2 = reward_labels_2 + reward_equal*0.5
            reward_input_combined_tf = tf.concat(axis=1, values=[self.main.reward_1_tf, self.main.reward_2_tf]) 
            reward_labels_combined_tf = tf.concat(axis=1, values=[reward_labels_1, reward_labels_2]) 
            self.reward_loss_tf = tf.nn.softmax_cross_entropy_with_logits(logits=reward_input_combined_tf, labels=reward_labels_combined_tf)
            # self.reward_loss_tf = -1 * tf.reduce_sum(tf.multiply(reward_labels_1, tf.log(self.main.reward_1_tf)) + tf.multiply(reward_labels_2, tf.log(self.main.reward_2_tf)))

        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        if self.reward_model:
            reward_grads_tf = tf.gradients(self.reward_loss_tf, self._vars('main/reward_model'))
        assert len(self._vars('main/Q')) == len(Q_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        if self.reward_model:
            self.reward_grads_vars_tf = zip(reward_grads_tf, self._vars('main/reward_model'))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))
        if self.reward_model:
            self.reward_grad_tf = flatten_grads(grads=reward_grads_tf, var_list=self._vars('main/reward_model'))
        # optimizers
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)
        if self.reward_model:
            self.reward_adam = MpiAdam(self._vars('main/reward_model'), scale_grad_by_procs=False)
        # polyak averaging
        if self.reward_model:
            self.main_vars = self._vars('main/Q') + self._vars('main/pi') + self._vars('main/reward_model')
            self.target_vars = self._vars('target/Q') + self._vars('target/pi') + self._vars('target/reward_model')
        else:
            self.main_vars = self._vars('main/Q') + self._vars('main/pi')
            self.target_vars = self._vars('target/Q') + self._vars('target/pi')

        if self.scope != 'sac0':
            if self.bc_loss_upper:
                if self.adversarial_loss:
                    discriminator_grads_tf = tf.gradients(self.discriminator_loss_tf, self._vars('main/discriminator'))
                    assert len(self._vars('main/discriminator')) == len(discriminator_grads_tf)
                    self.discriminator_grads_vars_tf = zip(discriminator_grads_tf, self._vars('main/discriminator'))
                    self.discriminator_grad_tf = flatten_grads(grads=discriminator_grads_tf, var_list=self._vars('main/discriminator'))
                    self.discriminator_adam = MpiAdam(self._vars('main/discriminator'), scale_grad_by_procs=False)
                    self.main_vars += self._vars('main/discriminator')
                    self.target_vars += self._vars('target/discriminator')

                if self.predictor_loss:
                    predictor_grads_tf = tf.gradients(self.predictor_loss_tf, self._vars('main/predictor'))
                    assert len(self._vars('main/predictor')) == len(predictor_grads_tf)
                    self.predictor_grads_vars_tf = zip(predictor_grads_tf, self._vars('main/predictor'))
                    self.predictor_grad_tf = flatten_grads(grads=predictor_grads_tf, var_list=self._vars('main/predictor'))
                    self.predictor_adam = MpiAdam(self._vars('main/predictor'), scale_grad_by_procs=False)
                    self.main_vars += self._vars('main/predictor')
                    self.target_vars += self._vars('target/predictor')
        else:
            if self.bc_loss:
                if self.adversarial_loss:
                    discriminator_grads_tf = tf.gradients(self.discriminator_loss_tf, self._vars('main/discriminator'))
                    assert len(self._vars('main/discriminator')) == len(discriminator_grads_tf)
                    self.discriminator_grads_vars_tf = zip(discriminator_grads_tf, self._vars('main/discriminator'))
                    self.discriminator_grad_tf = flatten_grads(grads=discriminator_grads_tf, var_list=self._vars('main/discriminator'))
                    self.discriminator_adam = MpiAdam(self._vars('main/discriminator'), scale_grad_by_procs=False)
                    self.main_vars += self._vars('main/discriminator')
                    self.target_vars += self._vars('target/discriminator')
    
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        if self.is_image_data:
            self.stats_vars += self._global_vars('p_stats') 
        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]
        if self.is_image_data:
            logs += [('stats_p/mean', np.mean(self.sess.run([self.p_stats.mean])))]
            logs += [('stats_p/std', np.mean(self.sess.run([self.p_stats.std])))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None



        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)
