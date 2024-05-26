import os
import sys
import time
import click
import numpy as np
import json
from mpi4py import MPI
import pickle
import resource
from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

sys.path.append('../preference_hrl/')
from her import make_sample_her_transitions
from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
import config
from rollout import RolloutWorkerOriginal
from util import mpi_fork

from subprocess import CalledProcessError

def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]


def train(policyList, rollout_worker, num_layers, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies, populate, reward_model, lower_reward_model,  dac, hac, hier, orpl, bc_reg, q_reg, T, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    params = config.DEFAULT_PARAMS
    is_multiple_env = params['is_multiple_env']
    num_upper_demos = params['num_upper_demos']
    env_name = params['env_name']

    if 'Rope' in env_name:
        demo_file = './data/robotic_rope_dataset.npz'
    elif 'Hollow' in env_name:
        demo_file = './data/hollow_dataset.npz'
    elif 'Bin' in env_name:
        demo_file = './data/bin_dataset.npz'
    elif 'Pick' in env_name:
        demo_file = './data/pick_dataset.npz'
    elif 'Maze' in env_name:
        demo_file = './data/maze_dataset.npz'
    elif 'kitchen' in env_name:
        demo_file = './data/kitchen_dataset.npz'
        # demo_file = './data/kitchen_dataset_microwave.npz'

    # For RPL experiments
    if 'Bin' in env_name:
        demo_file_rpl = './data/upper_bin_dataset.npz'
    elif 'Hollow' in env_name:
        demo_file_rpl = './data/upper_hollow_dataset.npz'
    elif 'Pick' in env_name:
        demo_file_rpl = './data/upper_pick_dataset.npz'
    elif 'Rope' in env_name:
        demo_file_rpl = './data/upper_rope_dataset.npz'
    elif 'Maze' in env_name:
        demo_file_rpl = './data/upper_maze_dataset.npz'
    elif 'kitchen' in env_name:
        demo_file_rpl = './data/upper_kitchen_dataset.npz'
        # demo_file_rpl = './data/upper_kitchen_dataset_microwave.npz'

    best_success_rate = -1
    best_success_epoch = 0

    if policyList[0].bc_loss:
        policyList[0].initDemoBuffer(demo_file)

    # For RPL experiments
    if orpl and len(policyList) > 1:
        policyList[1].initDemoBufferUpperMultiple(demo_file_rpl)

    import time
    start = time.time()
    print('Training...')
    for epoch in range(n_epochs):
        # logger.info("Training...")
        rollout_worker.clear_history()
        for _ in range(n_cycles):
            env_seed = np.random.randint(num_upper_demos)
            rollout_worker.generate_rollouts(env_seed, epoch, True)
            if len(policyList) == 1:
                for j in range(n_batches):
                    policyList[0].train()
                policyList[0].update_target_net()
            else:
                for i in range(len(policyList)):
                    if 1:#i != 0:
                        for j in range(n_batches):
                            policyList[i].train()
                        policyList[i].update_target_net()

        # test
        # logger.info("Testing")
        evaluator.clear_history()
        for test_num in range(n_test_rollouts):
            env_seed = np.random.randint(num_upper_demos)
            evaluator.generate_rollouts(env_seed, test_num, False)

        # record logs
        logger.record_tabular('1.  Epoch', epoch+1)
        if 'Rope' in env_name:
            logger.record_tabular('2.  Timesteps', epoch*105*(T**num_layers))
        else:
            logger.record_tabular('2.  Timesteps', epoch*(T**num_layers))
        logger.record_tabular('3.  Time elapsed ',int(time.time()-start))
        

        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()


        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and save_policies:
            if success_rate >= best_success_rate:
                best_success_rate = success_rate
                best_success_epoch = epoch
                evaluator.save_policy(best_policy_path)
            # else:
            #     evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            # logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        # logger.info("Best success rate so far ", best_success_rate, " In epoch number ", best_success_epoch)
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)

def launch(env, logdir, n_epochs, num_cpu, seed, populate, reward_model, lower_reward_model, num_hrl_layers, bc_loss, bc_loss_upper, 
    adversarial_loss, render, dac, hac, hier, raps, orpl, q_reg, flat, bc_reg, n_test_rollouts, n_batches , replay_strategy, reward_batch_size,
    override_params={}, cpu_index=-1, save_policies=True):
    # Fork for multi-CPU MPI implementation.
    logdir = './models/' + logdir
    if num_cpu > 1:
        try:
            whoami = mpi_fork(num_cpu,[])
            # whoami = mpi_fork(num_cpu,['--bind-to', 'core'])
        except CalledProcessError:
            # fancy version of mpi call failed, try simple version
            whoami = mpi_fork(num_cpu)

        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    elif cpu_index != -1:
        try:
            whoami = mpi_fork(num_cpu, cpu_index, [])
            # whoami = mpi_fork(num_cpu,['--bind-to', 'core'])
        except CalledProcessError:
            # fancy version of mpi call failed, try simple version
            whoami = mpi_fork(num_cpu, cpu_index)

        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)
    # resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['env_name'] = env
    params['num_hrl_layers'] = num_hrl_layers
    params['hac'] = hac
    params['hier'] = hier
    params['dac'] = dac
    params['replay_strategy'] = replay_strategy
    params['reward_batch_size'] = reward_batch_size
    if env in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)
    params = config.prepare_params(params)
    # config.log_params(params, logger=logger)

    num_layers = num_hrl_layers
    policy_save_interval = params['policy_save_interval']
    dims = config.configure_dims(params) # {'o': 10, 'u': 4, 'g': 3, 'info_is_success': 1}
    policyList = []
    # policy_file_pear = './models/maze_pear_0/policy_best.pkl'
    # with open(policy_file_pear, 'rb') as f:
    #     optimal_policy = pickle.load(f)
    optimal_policy = None
    for layer_num in range(num_layers):
        # if layer_num == 0:
        #     policy_file = './models/pick_pear_0/policy_best.pkl' #fetchPickAndPlace
        #     with open(policy_file, 'rb') as f:
        #         policy = pickle.load(f)[0]
        #     hrl_scope = 'ddpg' + str(layer_num)
        #     print(hrl_scope)
        #     params['hrl_scope'] = hrl_scope
        #     sample_func = config.configure_her(params)
        #     policy.sample_transitions = sample_func
        #     policy.bc_loss = bc_loss
        #     policy.bc_loss_upper = bc_loss_upper
        #     policy.sac = params['sac']
        #     policy.buffer.sample_transitions = sample_func
        #     buffer_temp = policy.access_global_demo_buffer()
        #     buffer_temp.sample_transitions = sample_func
        #     policyList.append(policy)
        # else:
        if layer_num == 0:
            hrl_scope = 'sac' + str(layer_num)
            params['hrl_scope'] = hrl_scope
            policy = config.configure_ddpg(dims=dims, params=params, hrl_scope = hrl_scope, 
                populate=populate, reward_model=reward_model, lower_reward_model=lower_reward_model, bc_loss=bc_loss, bc_loss_upper=bc_loss_upper, 
                adversarial_loss=adversarial_loss, hac=hac, hier=hier, dac=dac, bc_reg=bc_reg, q_reg=q_reg, optimal_policy=optimal_policy, lower_policy=None)
            policy.logdir = logdir
            policyList.append(policy)
        else:
            hrl_scope = 'sac' + str(layer_num)
            params['hrl_scope'] = hrl_scope
            policy = config.configure_ddpg(dims=dims, params=params, hrl_scope = hrl_scope, 
                populate=populate, reward_model=reward_model, lower_reward_model=lower_reward_model, bc_loss=bc_loss, bc_loss_upper=bc_loss_upper, 
                adversarial_loss=adversarial_loss, hac=hac, hier=hier, dac=dac, bc_reg=bc_reg, q_reg=q_reg, optimal_policy=optimal_policy, lower_policy=policyList[0])
            policy.logdir = logdir
            policyList.append(policy)
        # if num_layers > 1 and layer_num == 1:
        #     policyList[1].otherPolicy = policyList[0]
    
    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
        'hrl_imitation_loss': params['hrl_imitation_loss'],
        'adversarial_loss': adversarial_loss,
        'bc_loss': bc_loss,
        'bc_loss_upper': bc_loss_upper,
        'sac':params['sac'],
        'populate':populate,
        'reward_model':reward_model,
        'lower_reward_model':lower_reward_model,
        'hac':hac,
        'hier':hier,
        'dac':dac,
        'raps':raps,
        'predictor_loss':params['predictor_loss'],
        'is_multiple_env': params['is_multiple_env'],
        'discrete_maze': params['discrete_maze'],
        'upper_only_bc': params['upper_only_bc'],
        'num_upper_demos': params['num_upper_demos'],
        'render': render,
        'testingNow': 0,
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        #'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
        'hrl_imitation_loss': params['hrl_imitation_loss'],
        'adversarial_loss': adversarial_loss,
        'bc_loss': bc_loss,
        'bc_loss_upper': bc_loss_upper,
        'sac':params['sac'],
        'populate':populate,
        'reward_model':reward_model,
        'lower_reward_model':lower_reward_model,
        'hac':hac,
        'hier':hier,
        'dac':dac,
        'raps':raps,
        'predictor_loss':params['predictor_loss'],
        'is_multiple_env': params['is_multiple_env'],
        'discrete_maze': params['discrete_maze'],
        'upper_only_bc': params['upper_only_bc'],
        'num_upper_demos': params['num_upper_demos'],
        'render': render ,
        'testingNow': 1,
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps', 'is_image_data']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]


    rollout_worker = RolloutWorkerOriginal(params['make_env'], policyList, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)

    evaluator = RolloutWorkerOriginal(params['make_env'], policyList, dims, logger, **eval_params)
    evaluator.seed(rank_seed)
    # np.set_printoptions(suppress=True)
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    train(
        logdir=logdir, policyList=policyList, rollout_worker=rollout_worker, num_layers=num_layers,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=n_test_rollouts,
        n_cycles=params['n_cycles'], n_batches=n_batches,
        policy_save_interval=policy_save_interval, save_policies=save_policies, populate=populate, bc_reg=bc_reg, q_reg=q_reg,
        reward_model=reward_model, lower_reward_model=lower_reward_model, dac=dac, hac=hac, hier=hier, orpl=orpl, T = params['T'])

@click.command()
@click.option('--env', type=str, default='FetchMazeReach-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--logdir', type=str, default='fetchMaze_test', help='the path to where logs and policy pickles should go.')
@click.option('--n_epochs', type=int, default=20000, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--bc_loss', type=int, default=0, help='')
@click.option('--bc_loss_upper', type=int, default=0, help='')
@click.option('--adversarial_loss', type=int, default=0, help='')
@click.option('--num_hrl_layers', type=int, default=2, help='')
@click.option('--populate', type=int, default=0, help='')
@click.option('--reward_model', type=int, default=0, help='')
@click.option('--lower_reward_model', type=int, default=0, help='')
@click.option('--render', type=int, default=0, help='')
@click.option('--cpu_index', type=int, default=-1, help='')
@click.option('--dac', type=int, default=0, help='')
@click.option('--hac', type=int, default=0, help='')
@click.option('--raps', type=int, default=0, help='')
@click.option('--q_reg', type=int, default=0, help='')
@click.option('--bc_reg', type=int, default=0, help='')
@click.option('--hier', type=int, default=0, help='')
@click.option('--orpl', type=int, default=0, help='')
@click.option('--flat', type=int, default=0, help='')
@click.option('--n_batches', type=int, default=10, help='')
@click.option('--n_test_rollouts', type=int, default=0, help='')
@click.option('--reward_batch_size', type=int, default=50, help='')
@click.option('--replay_strategy', type=str, default='future', help='')

def main(**kwargs):
    launch(**kwargs)


if __name__ == '__main__':
    main()
