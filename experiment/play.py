import click
import numpy as np
import pickle
import sys

sys.path.append('../preference_hrl/')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

from baselines import logger
from baselines.common import set_global_seeds
import config
from rollout import RolloutWorkerOriginal

import time
@click.command()
# @click.argument('policy_file', type=str)
@click.option('--seed', type=int, default=0)
@click.option('--rollouts', type=int, default=100)
@click.option('--render', type=int, default=1)
@click.option('--dir', type=str, default='log')
@click.option('--policy', type=str, default=None)
def main(dir, seed, rollouts, render, policy):
    xx = time.time()
    set_global_seeds(seed)

    # Load policy.
    policyList = [None, None]
    if policy == None:
        with open('./models/'+dir+'/policy_best.pkl', 'rb') as f:
            policyList = pickle.load(f)
    else:
        with open('./models/'+dir+'/'+policy+'.pkl', 'rb') as f:
            policyList = pickle.load(f)
    env_name = policyList[0].info['env_name']

    # Prepare params.
    params = config.DEFAULT_PARAMS
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params['env_name'] = env_name
    if 'dac' in dir or 'flat' in dir or 'her' in dir:
        params['num_hrl_layers'] = 1
    else:
        params['num_hrl_layers'] = 2

    if ('pear_bc' not in dir and 'pear' in dir) or ('rpl_bc' not in dir and 'rpl' in dir) or 'dac' in dir:
        params['adversarial_loss'] = 1
    else:
        params['adversarial_loss'] = 0

    if 'pear' in dir:
        params['bc_loss'] = 1
        params['bc_loss_upper'] = 1
    elif 'rpl' in dir:
        params['bc_loss'] = 1
        params['bc_loss_upper'] = 1
    elif 'hier' in dir:
        params['bc_loss'] = 1
        params['bc_loss_upper'] = 0
    elif 'dac' in dir:
        params['bc_loss'] = 1
        params['bc_loss_upper'] = 0
    elif 'flat' in dir:
        params['bc_loss'] = 0
        params['bc_loss_upper'] = 0

    params = config.prepare_params(params)
    # config.log_params(params, logger=logger)

    dims = config.configure_dims(params)

    if params['env_name'] == 'GazeboWAMemptyEnv-v1':
        eval_params = {
            'exploit': True,
            'use_target_net': params['test_with_polyak'],
            'compute_Q': True,
            'rollout_batch_size': 1,
            #'render': bool(render),
        }

        for name in ['T', 'gamma', 'noise_eps', 'random_eps']:
            eval_params[name] = params[name]

        madeEnv = config.cached_make_env(params['make_env'])
        evaluator = RolloutWorker(madeEnv, params['make_env'], policy, dims, logger, **eval_params)
        evaluator.seed(seed)
    else:
        eval_params = {
            'exploit': True,
            'use_target_net': params['test_with_polyak'],
            'compute_Q': True,
            'rollout_batch_size': 1,
            'render': bool(render),
            'testingNow': 1,
            'hrl_imitation_loss': params['hrl_imitation_loss'],
            'adversarial_loss': params['adversarial_loss'],
            'bc_loss': params['bc_loss'],
            'bc_loss_upper': params['bc_loss_upper'],
            'sac':params['sac'],
            'is_multiple_env': params['is_multiple_env'],
            'upper_only_bc': params['upper_only_bc'],
            'num_upper_demos': params['num_upper_demos'],
        }

        for name in ['T', 'gamma', 'noise_eps', 'random_eps']:
            eval_params[name] = params[name]

        evaluator = RolloutWorkerOriginal(params['make_env'], policyList, dims, logger, **eval_params)
        evaluator.seed(seed)

    # Run evaluation.
    evaluator.clear_history()
    step_count = 0
    index_count = 0
    seed_count = -1
    for index in range(rollouts):
        seed_count = (seed_count+1)#%params['num_upper_demos']
        index_count += 1
        print('index',index+1,"out of",rollouts)
        env_seed = seed_count
        print('seed_sent',env_seed)
        is_successful_values, dist_values = evaluator.generate_rollouts(env_seed)
    print('Mean:', np.mean(dist_values))
    print('Std:', np.std(dist_values))

    # record logs
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()


if __name__ == '__main__':
    main()
