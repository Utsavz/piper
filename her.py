import numpy as np

def make_sample_her_transitions(replay_strategy, replay_k, reward_fun, hrl_scope, reward_model, lower_reward_model, hac, dac, bc_reg, q_reg, hier, env_name):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0
    if hrl_scope == 'sac0':
        reward_model = 0
    if lower_reward_model:
        reward_model = 1
    # if hrl_scope != 'sac0':
    #     future_p = 0
    # env_name = "kitchen"

    def _sample_her_transitions(episode_batch, batch_size_in_transitions, optimal_policy=None, lower_policy=None):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        # max_u = [0.25, 0.27, 0.]
        # action_offset = [1.3, 0.75, 0.42]
        if 'Reach' in env_name:
            max_u = [0.25, 0.27, 0.]
            action_offset = [1.3, 0.75, 0.42]
        elif 'kitchen' in env_name:
            max_u = [1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,0.88,0.01,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,0.75,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6]
            action_offset = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        else:
            max_u = [0.25, 0.27, 0.145]
            action_offset = [1.3, 0.75, 0.555]

        alpha = 0.000001
        beta = 0.0001

        rollout_batch_size = len(episode_batch['u'])
        # traj_len = len(episode_batch['u'][0])
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size) # selecting from among the rollout_batch_sizes(size of the buffer)

        epi_len = np.array([episode_batch['u'][x].shape[0] for x in episode_idxs])
        t_samples = np.random.uniform(0,epi_len-1,batch_size).astype(int)
        t_sam = t_samples

        if bc_reg:
            if lower_policy is not None:
                expert_data = lower_policy.get_demoBuffer_data()

        if replay_strategy == 'future':
            future_p = 1 - (1. / (1 + replay_k))
        else:  # 'replay_strategy' == 'none'
            future_p = 0
        
        transitions = {}        
        for key in episode_batch.keys():
            if key == 'info_is_success':
                continue
            transitions[key] = np.array([episode_batch[key][episode][sample] for episode,sample in zip(episode_idxs,t_samples)])

        if reward_model:
            reward_episode_idxs_1 = np.random.randint(0, rollout_batch_size, batch_size)
            reward_episode_idxs_2 = np.random.randint(0, rollout_batch_size, batch_size)
            transitions['reward_o_1'] = np.array([episode_batch['o'][episode][:-1,...] for episode in reward_episode_idxs_1])
            transitions['reward_o_2'] = np.array([episode_batch['o'][episode][:-1,...] for episode in reward_episode_idxs_2])
            transitions['reward_g_1'] = np.array([episode_batch['g'][episode] for episode in reward_episode_idxs_1])
            transitions['reward_g_2'] = np.array([episode_batch['g'][episode] for episode in reward_episode_idxs_2])
            transitions['reward_ag_1'] = np.array([episode_batch['ag'][episode][:-1,...] for episode in reward_episode_idxs_1])
            transitions['reward_ag_2'] = np.array([episode_batch['ag'][episode][:-1,...] for episode in reward_episode_idxs_2])
            transitions['reward_ag_2_1'] = np.array([episode_batch['ag_2'][episode] for episode in reward_episode_idxs_1])
            transitions['reward_ag_2_2'] = np.array([episode_batch['ag_2'][episode] for episode in reward_episode_idxs_2])
            transitions['reward_u_1'] = np.array([episode_batch['u'][episode] for episode in reward_episode_idxs_1])
            transitions['reward_u_2'] = np.array([episode_batch['u'][episode] for episode in reward_episode_idxs_2])

            if hrl_scope != 'sac0':
                transitions['reward_full_o_1'] = np.array([episode_batch['full_o'][episode] for episode in reward_episode_idxs_1])
                transitions['reward_full_o_2'] = np.array([episode_batch['full_o'][episode] for episode in reward_episode_idxs_2])
            
            # hindsight pl

            reward_epi_len_1 = np.array([transitions['reward_g_1'][x].shape[0] for x in range(len(transitions['reward_g_1']))])
            reward_epi_len_2 = np.array([transitions['reward_g_2'][x].shape[0] for x in range(len(transitions['reward_g_2']))])
            reward_goal_samples_1 = np.random.uniform(0,reward_epi_len_1-1,batch_size).astype(int)
            reward_goal_samples_2 = np.random.uniform(0,reward_epi_len_2-1,batch_size).astype(int)
            reward_goal_picker = np.random.uniform(0,2,batch_size).astype(int)
            reward_goal_samples = []
            for i, value in enumerate(reward_goal_picker):
                if value == 0:
                    reward_goal_samples.append(reward_goal_samples_1[i])
                else:
                    reward_goal_samples.append(reward_goal_samples_2[i])    
            reward_goal_samples = np.array(reward_goal_samples)
            reward_her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)[0]
            for i, idx in enumerate(reward_her_indexes):
                if reward_goal_picker[idx] == 0:
                    future_goal = transitions['reward_ag_1'][idx][reward_goal_samples[idx]]
                else:
                    future_goal = transitions['reward_ag_2'][idx][reward_goal_samples[idx]]
                transitions['reward_g_1'][idx] = future_goal
                transitions['reward_g_2'][idx] = future_goal

            reward_params_1 = {}
            reward_params_1['ag_2'] = transitions['reward_ag_2_1']
            reward_params_1['g'] = transitions['reward_g_1']
            reward_params_1['info'] = None
            reward_params_1['reward_type'] = 'dense'
            transitions['reward_r_1'] = reward_fun(**reward_params_1)
            transitions['reward_r_1'] = transitions['reward_r_1'].reshape(transitions['reward_r_1'].shape[0], transitions['reward_r_1'].shape[1], 1)

            reward_params_2 = {}
            reward_params_2['ag_2'] = transitions['reward_ag_2_2']
            reward_params_2['g'] = transitions['reward_g_2']
            reward_params_2['info'] = None
            reward_params_2['reward_type'] = 'dense'
            transitions['reward_r_2'] = reward_fun(**reward_params_2)
            transitions['reward_r_2'] = transitions['reward_r_2'].reshape(transitions['reward_r_2'].shape[0], transitions['reward_r_2'].shape[1], 1)

            transitions['reward_r_1'] = np.sum(np.array(transitions['reward_r_1']), axis=1)
            transitions['reward_r_2'] = np.sum(np.array(transitions['reward_r_2']), axis=1)

            # transitions['reward_r_1'] = np.array(transitions['reward_r_1'][:,-1,:])
            # transitions['reward_r_2'] = np.array(transitions['reward_r_2'][:,-1,:])

            # transitions['reward_r_1'] = np.sum(np.array([episode_batch['r'][episode] for episode in reward_episode_idxs_1]), axis=1)
            # transitions['reward_r_2'] = np.sum(np.array([episode_batch['r'][episode] for episode in reward_episode_idxs_2]), axis=1)
            # transitions['reward_r_1'] = (np.sum(np.array([episode_batch['r'][episode] for episode in reward_episode_idxs_1]), axis=1) > -traj_len)
            # transitions['reward_r_2'] = (np.sum(np.array([episode_batch['r'][episode] for episode in reward_episode_idxs_2]), axis=1) > -traj_len)

            # ret_1 = optimal_policy[1].get_Q_sac_reward_upper(transitions['reward_o_1'], transitions['reward_g_1'], transitions['reward_u_1'])
            # ret_2 = optimal_policy[1].get_Q_sac_reward_upper(transitions['reward_o_2'], transitions['reward_g_2'], transitions['reward_u_2'])
            # ret_1 = optimal_policy[0].get_Q_sac_reward_lower(transitions['reward_o_1'], transitions['reward_u_1'][...,:3])
            # ret_2 = optimal_policy[0].get_Q_sac_reward_lower(transitions['reward_o_2'], transitions['reward_u_2'][...,:3])
            # transitions['reward_r_1'] = np.mean(ret_1, axis=1)
            # transitions['reward_r_2'] = np.mean(ret_2, axis=1)

            if 'Fetch' in env_name:
                subgoal_supervision_1 = transitions['reward_u_1'][...,:3]
                subgoal_supervision_2 = transitions['reward_u_2'][...,:3]
            else:
                subgoal_supervision_1 = transitions['reward_u_1']
                subgoal_supervision_2 = transitions['reward_u_2']
            # if 'Rope' in str(self.envs[0]):
            #     u_temp = u[0].copy()
            #     u = self.rope_action_to_obs_encoder(u_temp)
            #     u = np.array(u).reshape(1,-1)
            # else:
                # if 'kitchen' in str(self.envs[0]):
                #     u = np.array(u[0]).reshape(1,-1)
                # else:

            if q_reg:
                action_supervision_1 = action_offset + (max_u * subgoal_supervision_1)
                action_supervision_2 = action_offset + (max_u * subgoal_supervision_2)
                if not lower_reward_model:
                    ret_1 = np.sum(lower_policy.get_Q_sac_reward_lower(transitions['reward_o_1'], action_supervision_1), axis=1)
                    ret_2 = np.sum(lower_policy.get_Q_sac_reward_lower(transitions['reward_o_2'], action_supervision_2), axis=1)
                    transitions['reward_r_1'] = transitions['reward_r_1'] + alpha * ret_1
                    transitions['reward_r_2'] = transitions['reward_r_2'] + alpha * ret_2

            if bc_reg:
                reward_params_expert_1 = {}
                reward_params_expert_1['ag_2'] = transitions['reward_full_o_1']
                reward_params_expert_1['g'] = np.tile(np.array(expert_data['o'])[:,:-1,:], (len(transitions['reward_full_o_1']),1,1))
                reward_params_expert_1['info'] = None
                reward_params_expert_1['reward_type'] = 'dense'
                transitions['reward_r_1'] += beta * np.mean(reward_fun(**reward_params_expert_1))

                reward_params_expert_2 = {}
                reward_params_expert_2['ag_2'] = transitions['reward_full_o_2']
                reward_params_expert_2['g'] = np.tile(np.array(expert_data['o'])[:,:-1,:], (len(transitions['reward_full_o_2']),1,1))
                reward_params_expert_2['info'] = None
                reward_params_expert_2['reward_type'] = 'dense'
                transitions['reward_r_2'] += beta * np.mean(reward_fun(**reward_params_expert_2))

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)[0] # Select future_p percent of all the indices(size = batch_size)
        future_offset = np.random.uniform(size=batch_size) * (epi_len - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        for i, idx in enumerate(episode_idxs[her_indexes]):
            if len(episode_batch['ag'][idx]) > future_t[i]:
                transitions['g'][her_indexes[i]] = episode_batch['ag'][idx][future_t[i]]

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        if future_p == 0:
            transitions['r'] = transitions['r'].reshape(-1)
        else:
            transitions['r'] = reward_fun(**reward_params)# + transitions['r'].reshape(-1)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)
        return transitions

    return _sample_her_transitions
