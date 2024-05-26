import numpy as np
import gym
# import gym_gazebo

import sys
sys.path.append('../preference_hrl/')

from baselines import logger
from ddpg import DDPG
from her import make_sample_her_transitions


DEFAULT_ENV_PARAMS = {
    'FetchReach-v1': {
        'n_cycles': 10,
    },
    'GazeboWAMemptyEnv-v2': {
        'n_cycles': 20,
    },
}


DEFAULT_PARAMS = {
    # env
    'max_u': 1.,  # Value by which symmetric actions are scales(actor outputs a tanh(-1 to 1))
    # 'action_offset': 0., # Offset(==mid point of the extremes) added to output of actor
    # sac
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 512,  # number of neurons in each hidden layers
    'network_class': 'actor_critic:ActorCritic',
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'buffer_size': int(1E7),  # for experience replay
    'polyak': 0.8,  # polyak averaging coefficient
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'scope': 'sac',  # can be tweaked for testing
    'relative_goals': False,
    # training
    'n_cycles': 1,  # per epoch
    'rollout_batch_size': 1,  # per mpi thread
    'n_batches': 10,  # training batches per cycle
    'batch_size': 1024,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'reward_batch_size': 50,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'n_test_rollouts': 0,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'test_with_polyak': False,  # run test episodes with the target network
    # exploration
    'random_eps': 0.2,  # percentage of time a random action is taken
    'alpha': 0.05, # weightage parameter for SAC
    'noise_eps': 0.05,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
    # normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values
    'bc_loss': 1, # whether or not to use the behavior cloning loss as an auxilliary loss
    'bc_loss_upper': 0, # whether or not to use the behavior cloning loss as an auxilliary loss at upper level
    'q_filter': 1, # whether or not a Q value filter should be used on the Actor outputs
    'clutter_reward': 0, # whether or not a lesser clutter near the arm yields a reward, 1=yes, 0=no
    'policy_save_interval': 5000, # the interval with which policy pickles are saved.
    'num_demo': 100, # number of expert demo episodes
    'clutter_num': 60, # Clutter Number around the object
    'adversarial_loss': 0, # Training upper levels via adversarial loss
    'hrl_imitation_loss': 0, # Imitation Learning loss for upper layers
    'num_hrl_layers': 2, # Number of hierarchical layers
    'is_image_data': 0, # Whether image data is used
    'sac': 1, # sac=1(SAC used), sac=0(ddpg used)
    'populate': 0,
    'reward_model': 0,
    'lower_reward_model': 0,
    'predictor_loss': 0,
    'is_multiple_env': 1,
    'num_upper_demos': 1,
    'upper_only_bc': 0,
    'discrete_maze': 0,
}


CACHED_ENVS = {}
# from gym_maze.envs.maze_env import MazeEnv

def cached_make_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if make_env not in CACHED_ENVS:
        env = make_env()
        # CACHED_ENVS[make_env] = env
    # return CACHED_ENVS[make_env]
    return env

def prepare_params(kwargs):
    # SAC params
    ddpg_params = dict()
    env_name = kwargs['env_name']

    def make_env():
        # if env_name.split('-')[0] == 'MazeEnv':
        if kwargs['discrete_maze']:
            m_size = int(env_name.split('-')[1])
            return MazeEnv(maze_size=(m_size,m_size))
        else:
            return gym.make(env_name)
    kwargs['make_env'] = make_env
    tmp_env = cached_make_env(kwargs['make_env'])
    # assert hasattr(tmp_env, '_max_episode_steps')
    # kwargs['T'] = tmp_env._max_episode_steps
    num_hrl_layers = kwargs['num_hrl_layers']

    if "FetchMazeReach" in env_name:
        if num_hrl_layers == 2:
            kwargs['T'] = 15
        else:
            kwargs['T'] = 225

    elif "Bin" in env_name:
        if num_hrl_layers == 2:
            kwargs['T'] = 8
        else:
            kwargs['T'] = 64

    elif "Hollow" in env_name:
        if num_hrl_layers == 2:
            kwargs['T'] = 10
        else:
            kwargs['T'] = 100

    elif "FetchPickAndPlace" in env_name:
        if num_hrl_layers == 2:
            kwargs['T'] = 7
        else:
            kwargs['T'] = 49

    elif "Rope" in env_name:
        if num_hrl_layers == 2:
            kwargs['T'] = 5
        else:
            kwargs['T'] = 25

    elif "kitchen" in env_name:
        if num_hrl_layers == 2:
            kwargs['T'] = 15
        else:
            kwargs['T'] = 225

    # if "Ant" in env_name or "Swimmer" in env_name:
    #     if num_hrl_layers == 2:
    #         kwargs['T'] = 80
    #     else:
    #         kwargs['T'] = 400

    # if "Point" in env_name:
    #     if num_hrl_layers == 2:
    #         kwargs['T'] = 15
    #     else:
    #         kwargs['T'] = 225

    tmp_env.reset()
    kwargs['max_u'] = np.array(kwargs['max_u']) if isinstance(kwargs['max_u'], list) else kwargs['max_u']
    kwargs['gamma'] = 0.99#1. - 1. / kwargs['T']
    if 'lr' in kwargs:
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['Q_lr'] = kwargs['lr']
        del kwargs['lr']
    for name in ['buffer_size', 'hidden', 'layers',
                 'network_class',
                 'polyak',
                 'batch_size', 'Q_lr', 'pi_lr',
                 'norm_eps', 'norm_clip', 'max_u',
                 'action_l2', 'clip_obs', 'scope', 'relative_goals', 'is_image_data', 'env_name']:
        ddpg_params[name] = kwargs[name]
        kwargs['_' + name] = kwargs[name]
        # del kwargs[name]
    ddpg_params['reward_batch_size'] = kwargs['reward_batch_size']
    kwargs['ddpg_params'] = ddpg_params

    return kwargs


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def configure_her(params):
    env = cached_make_env(params['make_env'])
    env.reset()
    # clutterNumber = 0
    def reward_fun(ag_2, g, info, reward_type='sparse'):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info, reward_type=reward_type)

    # Prepare configuration for HER.
    her_params = {
        'reward_fun': reward_fun,
        # 'reward_fun_clutter': reward_fun_clutter,
    }
    # her_params['clutter_reward'] = params['clutter_reward']
    if 'new_replay_strategy' in params:
        params['replay_strategy'] = params['new_replay_strategy']
    for name in ['replay_strategy', 'replay_k']:
        her_params[name] = params[name]
        params['_' + name] = her_params[name]
    her_params['hrl_scope'] = params['hrl_scope']
    her_params['reward_model'] = params['reward_model']
    her_params['lower_reward_model'] = params['lower_reward_model']
    her_params['hac'] = params['hac']
    her_params['hier'] = params['hier']
    her_params['dac'] = params['dac']
    her_params['bc_reg'] = params['bc_reg']
    her_params['q_reg'] = params['q_reg']
    her_params['env_name'] = params['env_name']
    sample_her_transitions = make_sample_her_transitions(**her_params)

    return sample_her_transitions


def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b


def configure_ddpg(dims, params, hrl_scope, populate, reward_model, lower_reward_model, bc_loss, bc_loss_upper, adversarial_loss,
    lower_policy=None , optimal_policy=None, hac=0, hier=0, dac=0, bc_reg=0, q_reg=0, reuse=False, use_mpi=True, clip_return=True):
    params['reward_model'] = reward_model
    params['lower_reward_model'] = lower_reward_model
    params['bc_reg'] = bc_reg
    params['q_reg'] = q_reg
    sample_her_transitions = configure_her(params)
    # Extract relevant parameters.
    ddpg_params = params['ddpg_params']
    gamma = params['gamma']
    input_dims = dims.copy()
    # SAC agent
    env = cached_make_env(params['make_env'])
    env.reset()
    ddpg_params.update({'input_dims': input_dims,  # agent takes an input observations
                        'T': params['T'],
                        'clip_pos_returns': True,  # clip positive returns
                        'clip_return': (1. / (1. - gamma)) if clip_return else np.inf,  # max abs of return
                        'rollout_batch_size': params['rollout_batch_size'],
                        'subtract_goals': simple_goal_subtract,
                        'sample_transitions': sample_her_transitions,
                        'alpha': params['alpha'],
                        'gamma': params['gamma'],
                        'sac': params['sac'],
                        'populate': populate,
                        'reward_model':reward_model,
                        'lower_reward_model': lower_reward_model,
                        'predictor_loss': params['predictor_loss'],
                        'bc_loss': bc_loss,
                        'bc_loss_upper': bc_loss_upper,
                        'dac': dac,
                        'hac': hac,
                        'bc_reg': bc_reg,
                        'q_reg': q_reg,
                        'hier': hier,
                        'q_filter': params['q_filter'],
                        'num_demo': params['num_demo'],
                        'clutter_reward': params['clutter_reward'],
                        'hrl_imitation_loss': params['hrl_imitation_loss'],
                        'adversarial_loss': adversarial_loss,
                        'discrete_maze': params['discrete_maze'],
                        'is_image_data': params['is_image_data'],
                        'is_multiple_env': params['is_multiple_env'],
                        'upper_only_bc': params['upper_only_bc'],
                        'num_upper_demos': params['num_upper_demos'],
                        'num_hrl_layers': params['num_hrl_layers'],
                        'optimal_policy': optimal_policy,
                        'lower_policy': lower_policy,
                        })
    ddpg_params['info'] = {
        'env_name': params['env_name'],
    }
    ddpg_params['scope'] = hrl_scope
    policy = DDPG(reuse=reuse, **ddpg_params, use_mpi=use_mpi)
    return policy


def configure_dims(params):
    env = cached_make_env(params['make_env'])
    env.reset()
    if 'kitchen' in params['env_name']:
        obs, _, _, info = env.step(env.action_space.sample())
    elif 'Rope' in params['env_name']:
        obs, _, _, info = env.step_rope(env.action_space.sample())
    else:
        obs, _, _, info = env.step_maze(env.action_space.sample())
    action_shape = env.action_space.shape
    if params['discrete_maze']:
        action_size = 5
    else:
        action_size = action_shape[0]
    dims = {
        'o': obs['observation'].shape[0],
        'u': action_size,
        'g': obs['desired_goal'].shape[0],
    }
    if DEFAULT_PARAMS['is_multiple_env'] == 1:
        dims['env_indexes'] = 1

    # for key, value in info.items():
    #     value = np.array(value)
    #     if value.ndim == 0:
    #         value = value.reshape(1)
    #     dims['info_{}'.format(key)] = value.shape[0]
    return dims
