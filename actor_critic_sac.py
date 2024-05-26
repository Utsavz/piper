import tensorflow as tf
import numpy as np
from util import store_args, nn, features, featuresDQN13, featuresDQN15, featuresDoom, convDQN15, denseDQN15

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPS = 1e-8

class ActorCriticSac:
    @store_args
    def __init__(self, inputs_tf, next_inputs_tf, adversarial_inputs_tf, adversarial_loss, dimo, dimg, dimu, reward_model, 
                    max_u, o_stats, g_stats, p_stats, hidden, layers, predictor_loss=None, **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.predictor_loss = predictor_loss
        self.adversarial_loss = adversarial_loss
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']
        self.o_2_tf = next_inputs_tf['o']
        self.g_2_tf = next_inputs_tf['g']

        if self.reward_model:
            self.reward_o_1_tf = inputs_tf['reward_o_1']
            self.reward_o_2_tf = inputs_tf['reward_o_2']
            self.reward_g_1_tf = inputs_tf['reward_g_1']
            self.reward_g_2_tf = inputs_tf['reward_g_2']
            self.reward_u_1_tf = inputs_tf['reward_u_1']
            self.reward_u_2_tf = inputs_tf['reward_u_2']
        if self.is_image_data:
            self.p_tf = inputs_tf['p']
            self.p_2_tf = next_inputs_tf['p']
        weights_data = adversarial_inputs_tf #To change name later
        self.batch_size = 1024
        self.adversarial_batch_size = kwargs['adversarial_batch_size']
        self.predict_batch_size = kwargs['predict_batch_size']
        self.adversarial_predict_batch_size = kwargs['adversarial_predict_batch_size']
        self.demo_batch_size = kwargs['demo_batch_size']
        self.penulti_linear = 512
        self.dim_image = 30

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        o_2 = self.o_stats.normalize(self.o_2_tf)
        g_2 = self.g_stats.normalize(self.g_2_tf)
        if self.reward_model:
            self.reward_o_1_tf = self.o_stats.normalize(self.reward_o_1_tf)
            self.reward_o_2_tf = self.o_stats.normalize(self.reward_o_2_tf)
            self.reward_g_1_tf = self.g_stats.normalize(self.reward_g_1_tf)
            self.reward_g_2_tf = self.g_stats.normalize(self.reward_g_2_tf)
        if self.is_image_data:
            p = self.p_stats.normalize(self.p_tf)
            p_2 = self.p_stats.normalize(self.p_2_tf)
            pos = p[:,:3]
            gates = p[:,124:]
            pos_2 = p_2[:,:3]
        
        if self.is_image_data:
            input_pi_reshaped = tf.reshape(o, [-1, self.dim_image, self.dim_image, 1])
            input_pi_2_reshaped = tf.reshape(o_2, [-1, self.dim_image, self.dim_image, 1])
        else:
            input_pi = tf.concat(axis=1, values=[o, g])  # for actor
            input_pi_2 = tf.concat(axis=1, values=[o_2, g_2])  # for actor


        # Networks.
        # Actor/Generator network
        with tf.compat.v1.variable_scope('pi'):
            if self.is_image_data:
                # Image CNN Network
                conv_out = convDQN15(input_pi_reshaped, feature_size=512)
                dense_out = denseDQN15(conv_out, self.penulti_linear, feature_size=64)
                conv_out_2 = convDQN15(input_pi_2_reshaped, feature_size=512, reuse=True)
                dense_out_2 = denseDQN15(conv_out_2, self.penulti_linear, feature_size=64, reuse=True)
                input_pi = tf.concat(axis=1, values=[pos, g, dense_out])
                input_pi_2 = tf.concat(axis=1, values=[pos_2, g_2, dense_out_2])
                
            net = nn(input_pi, [self.hidden] * self.layers, last_activation=tf.nn.relu)
            mu = tf.layers.dense(net, self.dimu, activation=None, name="dense_mu")
            log_std = tf.layers.dense(net, self.dimu, activation=None, name="dense_std")
            log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = tf.exp(log_std)
            pi = mu + tf.random_normal(tf.shape(mu)) * std
            logp_pi = self.gaussian_likelihood(pi, mu, log_std)
            self.mu_tf, self.pi_tf, self.logp_pi_tf = self.apply_squashing_func(mu, pi, logp_pi)

            net_2 = nn(input_pi_2, [self.hidden] * self.layers, last_activation=tf.nn.relu, reuse=True)
            mu_2 = tf.layers.dense(net_2, self.dimu, activation=None, name="dense_mu", reuse=True)
            log_std_2 = tf.layers.dense(net_2, self.dimu, activation=None, name="dense_std", reuse=True)
            log_std_2 = tf.clip_by_value(log_std_2, LOG_STD_MIN, LOG_STD_MAX)
            std_2 = tf.exp(log_std_2)
            pi_2 = mu_2 + tf.random_normal(tf.shape(mu_2)) * std_2
            logp_pi_2 = self.gaussian_likelihood(pi_2, mu_2, log_std_2)
            self.mu_2_tf, self.pi_2_tf, self.logp_pi_2_tf = self.apply_squashing_func(mu_2, pi_2, logp_pi_2)


        with tf.compat.v1.variable_scope('Q1'):
            # for policy training using Q1
            if self.is_image_data:
                conv_out = convDQN15(input_pi_reshaped, feature_size=512)
                dense_out = denseDQN15(conv_out, self.penulti_linear, feature_size=64)
                input_Q = tf.concat(axis=1, values=[pos, g, dense_out, self.pi_tf / self.max_u])
            else:
                input_Q = tf.concat(axis=1, values=[input_pi, self.pi_tf / self.max_u]) #actions from the policy
            self.Q1_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1]) # define the actor
            # for critic training
            if self.is_image_data:
                input_Q = tf.concat(axis=1, values=[pos, g, dense_out, self.u_tf / self.max_u])
            else:
                input_Q = tf.concat(axis=1, values=[input_pi, self.u_tf / self.max_u]) #actions from the buffer
            self._input_Q1 = input_Q  # exposed for tests
            self.Q1_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True) # define the critic

        with tf.compat.v1.variable_scope('Q2'):
            # for policy training using Q2
            if self.is_image_data:
                conv_out = convDQN15(input_pi_reshaped, feature_size=512)
                dense_out = denseDQN15(conv_out, self.penulti_linear, feature_size=64)
                input_Q = tf.concat(axis=1, values=[pos, g, dense_out, self.pi_tf / self.max_u])
            else:
                input_Q = tf.concat(axis=1, values=[input_pi, self.pi_tf / self.max_u]) #actions from the policy
            self.Q2_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1]) # define the actor
            # for critic training
            if self.is_image_data:
                input_Q = tf.concat(axis=1, values=[pos, g, dense_out, self.u_tf / self.max_u])
            else:
                input_Q = tf.concat(axis=1, values=[input_pi, self.u_tf / self.max_u]) #actions from the buffer
            self._input_Q1 = input_Q  # exposed for tests
            self.Q2_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True) # define the critic

        if self.reward_model:
            with tf.compat.v1.variable_scope('reward_model'):
                # to train reward model
                input_reward_og_1 = tf.concat(axis=2, values=[self.reward_o_1_tf, self.reward_g_1_tf]) 
                input_reward_og_2 = tf.concat(axis=2, values=[self.reward_o_2_tf, self.reward_g_2_tf])

                input_og_1_tf = tf.concat(axis=2, values=[input_reward_og_1, self.reward_u_1_tf / self.max_u])
                input_og_2_tf = tf.concat(axis=2, values=[input_reward_og_2, self.reward_u_2_tf / self.max_u])
                input_reward_og_pred_2 = tf.concat(axis=1, values=[input_pi, self.u_tf / self.max_u])

                self._input_og_1_tf = input_og_1_tf  # exposed for tests
                self._input_og_2_tf = input_og_2_tf  # exposed for tests
                self.reward_1_tf = tf.reduce_sum(tf.tanh(nn(input_og_1_tf, [self.hidden] * self.layers + [1])), axis=1) # define reward model
                self.reward_2_tf = tf.reduce_sum(tf.tanh(nn(input_og_2_tf, [self.hidden] * self.layers + [1], reuse=True)), axis=1) # define reward model
                self.reward_pred_tf = tf.tanh(nn(input_reward_og_pred_2, [self.hidden] * self.layers + [1], reuse=True))

        if self.scope != 'sac0':
            if self.predictor_loss and self.adversarial_loss:
                with tf.compat.v1.variable_scope('discriminator'):
                    discriminator_o = input_pi
                    discriminator_u_1 = self.pi_tf[:self.batch_size - 2 * self.adversarial_predict_batch_size]
                    discriminator_u_2 = self.u_tf[self.batch_size - 2 * self.adversarial_predict_batch_size:self.batch_size - self.adversarial_predict_batch_size]
                    discriminator_u_3 = self.pi_tf[self.batch_size - self.adversarial_predict_batch_size:]
                    discriminator_u = tf.concat(axis=0, values=[discriminator_u_1, discriminator_u_2, discriminator_u_3])
                    input_discriminator = tf.concat(axis=1, values=[discriminator_o, discriminator_u])
                    
                    self.discriminator_pred_tf = tf.nn.sigmoid(nn(input_discriminator, [self.hidden] * self.layers + [1]))
                    input_pi_discriminator = tf.concat(axis=1, values=[discriminator_o, self.pi_tf])
                    self.discriminator_pi_pred_tf = tf.nn.sigmoid(nn(input_pi_discriminator, [self.hidden] * self.layers + [1], reuse=True))

                with tf.compat.v1.variable_scope('predictor'):
                    predictor_o = input_pi
                    predictor_u_1 = self.pi_tf[:self.batch_size - self.predict_batch_size]
                    predictor_u_2 = self.u_tf[self.batch_size - self.predict_batch_size:]
                    predictor_u = tf.concat(axis=0, values=[predictor_u_1, predictor_u_2])
                    input_predictor = tf.concat(axis=1, values=[predictor_o, predictor_u])

                    self.predictor_pred_tf = tf.nn.sigmoid(nn(input_predictor, [self.hidden] * self.layers + [1]))
                    input_pi_predictor = tf.concat(axis=1, values=[predictor_o, self.pi_tf])
                    self.predictor_pi_pred_tf = tf.nn.sigmoid(nn(input_pi_predictor, [self.hidden] * self.layers + [1], reuse=True))

            elif self.predictor_loss:
                with tf.compat.v1.variable_scope('predictor'):
                    predictor_o = input_pi
                    predictor_u_1 = self.pi_tf[:self.batch_size - self.predict_batch_size]
                    predictor_u_2 = self.u_tf[self.batch_size - self.predict_batch_size:]
                    predictor_u = tf.concat(axis=0, values=[predictor_u_1, predictor_u_2])
                    input_predictor = tf.concat(axis=1, values=[predictor_o, predictor_u])

                    self.predictor_pred_tf = tf.nn.sigmoid(nn(input_predictor, [self.hidden] * self.layers + [1]))
                    input_pi_predictor = tf.concat(axis=1, values=[predictor_o, self.pi_tf])
                    self.predictor_pi_pred_tf = tf.nn.sigmoid(nn(input_pi_predictor, [self.hidden] * self.layers + [1], reuse=True))

            elif self.adversarial_loss:
                with tf.compat.v1.variable_scope('discriminator'):
                    discriminator_o = input_pi
                    discriminator_u_1 = self.pi_tf[:self.batch_size - self.adversarial_batch_size]
                    discriminator_u_2 = self.u_tf[self.batch_size - self.adversarial_batch_size:]
                    discriminator_u = tf.concat(axis=0, values=[discriminator_u_1, discriminator_u_2])
                    input_discriminator = tf.concat(axis=1, values=[discriminator_o, discriminator_u])

                    self.discriminator_pred_tf = tf.nn.sigmoid(nn(input_discriminator, [self.hidden] * self.layers + [1]))
                    input_pi_discriminator = tf.concat(axis=1, values=[discriminator_o, self.pi_tf])
                    self.discriminator_pi_pred_tf = tf.nn.sigmoid(nn(input_pi_discriminator, [self.hidden] * self.layers + [1], reuse=True))
        else:
            if self.adversarial_loss:
                with tf.compat.v1.variable_scope('discriminator'):
                    discriminator_o = input_pi
                    discriminator_u_1 = self.pi_tf[:self.batch_size - self.adversarial_batch_size]
                    discriminator_u_2 = self.u_tf[self.batch_size - self.adversarial_batch_size:]
                    discriminator_u = tf.concat(axis=0, values=[discriminator_u_1, discriminator_u_2])
                    input_discriminator = tf.concat(axis=1, values=[discriminator_o, discriminator_u])

                    self.discriminator_pred_tf = (nn(input_discriminator, [self.hidden] * self.layers + [1]))
                    input_pi_discriminator = tf.concat(axis=1, values=[discriminator_o, self.pi_tf])
                    self.discriminator_pi_pred_tf = (nn(input_pi_discriminator, [self.hidden] * self.layers + [1], reuse=True))



    def nn(self,input, layers_sizes, weights_data=None, reuse=None, flatten=False, last_activation=None, name=""):
        """Creates a simple neural network
        """
        for i, size in enumerate(layers_sizes):
            if i < len(layers_sizes) - 1:
                activation = tf.nn.relu
            else:
                activation = None
            with tf.compat.v1.variable_scope('layer'+str(i)):
                W1 = tf.Variable(weights_data[i][0], name=name + '_' + str(i)+'weight')
                b1 = tf.Variable(weights_data[i][1], name=name + '_' + str(i)+'bias')
                h1 = tf.add(tf.matmul(input,W1),b1)
            input = h1
            if activation:
                input = activation(input)
        if flatten:
            assert layers_sizes[-1] == 1
            input = tf.reshape(input, [-1])
        return input

    def apply_squashing_func(self, mu, pi, logp_pi):
        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        # logp_pi -= tf.reduce_sum(tf.log(self.clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
        logp_pi -= tf.reduce_sum(2*(np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)
        mu = tf.tanh(mu)
        pi = tf.tanh(pi)

        return mu, pi, logp_pi

    def gaussian_likelihood(self, x, mu, log_std):
        pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    # def clip_but_pass_gradient(self, x, l=-1., u=1.):
    #     clip_up = tf.cast(x > u, tf.float32)
    #     clip_low = tf.cast(x < l, tf.float32)
    #     return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)