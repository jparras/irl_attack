from baselines.trpo_mpi import trpo_mpi as trainer
import argparse
from generate_env import generate_env
from eval_policy import PolicyEvaluator
import tensorflow as tf
import baselines.common.tf_util as U
import tensorflow.contrib as tfc
import gym
from baselines.common.distributions import make_pdtype
import pickle


class MeanEmbedding:
    def __init__(self, input_ph, hidden_sizes, nr_obs, dim_obs, layer_norm=False):

        num_layers = len(hidden_sizes)
        reshaped_input = tf.reshape(input_ph, shape=(-1, int(dim_obs)))
        data_input_layer = tf.slice(reshaped_input, [0, 0], [-1, dim_obs - 2]) # Erase two last values of each observation (these are flags)
        valid_input_layer = tf.slice(reshaped_input, [0, dim_obs - 2], [-1, 1]) # Used to obtain valid indexes
        valid_indices = tf.where(tf.cast(valid_input_layer, dtype=tf.bool))[:, 0:1]
        valid_data = tf.gather_nd(data_input_layer, valid_indices)

        last_out = valid_data

        if num_layers > 0:
            for i in range(num_layers):
                last_out = tf.layers.dense(last_out, hidden_sizes[i], name="fc%i" % (i + 1),
                                           kernel_initializer=U.normc_initializer(1.0))
                if layer_norm:
                    last_out = tfc.layers.layer_norm(last_out)
                last_out = tf.nn.relu(last_out)

            fc_out = last_out

            last_out_scatter = tf.scatter_nd(valid_indices, fc_out,
                                             shape=tf.cast(
                                                 [tf.shape(data_input_layer)[0], tf.shape(fc_out)[1]],
                                                 tf.int64)) # Reciprocal to tf.gather_nd (replaces values in tensor)

            reshaped_output = tf.reshape(last_out_scatter, shape=(-1, nr_obs, hidden_sizes[-1]))

        else:
            reshaped_output = tf.reshape(data_input_layer, shape=(-1, nr_obs, dim_obs - 2))

        reshaped_nr_obs_var = tf.reshape(valid_input_layer, shape=(-1, nr_obs, 1))

        n = tf.maximum(tf.reduce_sum(reshaped_nr_obs_var, axis=1, name="nr_agents_test"), 1)  # Use maximum for n>=1

        last_out_sum = tf.reduce_sum(reshaped_output, axis=1)
        last_out_mean = tf.divide(last_out_sum, n)

        self.me_out = last_out_mean


class MlpPolicy_Multi_Mean_Embedding(object):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self.layer_norm = False
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, feat_size, gaussian_fixed_var=True, index=None):

        num_hid_layers = len(hid_size)
        n_mean_embs = len(ob_space.dim_mean_embs)
        mean_emb_0 = ob_space.dim_mean_embs[0]
        mean_emb_1 = ob_space.dim_mean_embs[1]
        nr_obs_0 = mean_emb_0[0]
        dim_obs_0 = mean_emb_0[1]

        nr_obs_1 = mean_emb_1[0]
        dim_obs_1 = mean_emb_1[1]

        dim_flat_obs = ob_space.dim_flat_o

        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)

        if index is None:
            ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=(None,) + ob_space.shape)
        else:
            ob = U.get_placeholder(name="ob_" + str(index), dtype=tf.float32, shape=(None,) + ob_space.shape)

        mean_emb_0_input_layer = tf.slice(ob, [0, 0], [-1, nr_obs_0 * dim_obs_0])
        mean_emb_1_input_layer = tf.slice(ob, [0, nr_obs_0 * dim_obs_0], [-1, nr_obs_1 * dim_obs_1])
        flat_feature_input_layer = tf.slice(ob, [0, nr_obs_0 * dim_obs_0 + nr_obs_1 * dim_obs_1], [-1, dim_flat_obs])

        with tf.variable_scope('vf'):
            with tf.variable_scope('me_rec'):
                me_v_rec = MeanEmbedding(mean_emb_0_input_layer, feat_size[0], nr_obs_0, dim_obs_0)
            with tf.variable_scope('me_local'):
                me_v_local = MeanEmbedding(mean_emb_1_input_layer, feat_size[1], nr_obs_1, dim_obs_1)
            last_out = tf.concat([me_v_rec.me_out, me_v_local.me_out, flat_feature_input_layer], axis=1)
            for i in range(num_hid_layers):
                last_out = tf.layers.dense(last_out, hid_size[i], name="fc%i" % (i + 1),
                                           kernel_initializer=U.normc_initializer(1.0))
                if self.layer_norm:
                    last_out = tfc.layers.layer_norm(last_out)
                last_out = tf.nn.relu(last_out)

            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        with tf.variable_scope('pol'):
            with tf.variable_scope('me_rec'):
                me_pi_rec = MeanEmbedding(mean_emb_0_input_layer, feat_size[0], nr_obs_0, dim_obs_0)
            with tf.variable_scope('me_local'):
                me_pi_local = MeanEmbedding(mean_emb_1_input_layer, feat_size[1], nr_obs_1, dim_obs_1)
            last_out = tf.concat([me_pi_rec.me_out, me_pi_local.me_out, flat_feature_input_layer], axis=1)
            for i in range(num_hid_layers):
                last_out = tf.layers.dense(last_out, hid_size[i], name="fc%i" % (i + 1),
                                           kernel_initializer=U.normc_initializer(1.0))
                if self.layer_norm:
                    last_out = tfc.layers.layer_norm(last_out)
                last_out = tf.nn.relu(last_out)

            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final', activation=tf.nn.sigmoid,
                                       kernel_initializer=U.normc_initializer(0.1))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2],
                                         initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final',
                                          kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        #ac = U.switch(stochastic, tf.nn.sigmoid(last_out), tf.nn.sigmoid(last_out))
        self._act = U.function([stochastic, ob], [ac, self.vpred])
        #self._me = U.function([ob], [me])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob)
        return ac1, vpred1

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []

    def step(self, ob, stochastic):
        ac1, vpred1 = self._act(stochastic, ob)
        return ac1, vpred1, None, None  # The step method is slightly different to act method, included for compatibility

    def save(self, save_path):
        U.save_state(save_path, sess=tf.get_default_session())

    def load(self, load_path):
        U.load_state(load_path, sess=tf.get_default_session())


class MlpPolicy_No_Mean_Emmbedding(object):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self.layer_norm = False
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, feat_size, gaussian_fixed_var=True, index=None):

        num_hid_layers = len(hid_size)

        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)

        if index is None:
            ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=(None,) + ob_space.shape)
        else:
            ob = U.get_placeholder(name="ob_" + str(index), dtype=tf.float32, shape=(None,) + ob_space.shape)

        with tf.variable_scope('vf'):  # Value NN
            last_out = ob
            for i in range(num_hid_layers):
                last_out = tf.layers.dense(last_out, hid_size[i], name="fc%i" % (i + 1),
                                           kernel_initializer=U.normc_initializer(1.0))
                if self.layer_norm:
                    last_out = tfc.layers.layer_norm(last_out)
                last_out = tf.nn.relu(last_out)

            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        with tf.variable_scope('pol'):  # Policy NN
            last_out = ob
            for i in range(num_hid_layers):
                last_out = tf.layers.dense(last_out, hid_size[i], name="fc%i" % (i + 1),
                                           kernel_initializer=U.normc_initializer(1.0))
                if self.layer_norm:
                    last_out = tfc.layers.layer_norm(last_out)
                last_out = tf.nn.relu(last_out)

            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final',
                                       activation=tf.nn.sigmoid, kernel_initializer=U.normc_initializer(0.1))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2],
                                         initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final',
                                          kernel_initializer=U.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        #ac = U.switch(stochastic, tf.nn.sigmoid(last_out), tf.nn.sigmoid(last_out))
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob)
        return ac1, vpred1

    def step(self, ob, stochastic):
        ac1, vpred1 = self._act(stochastic, ob)
        return ac1, vpred1, None, None  # The step method is slightly different to act method, included for compatibility

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []

    def save(self, save_path):
        U.save_state(save_path, sess=tf.get_default_session())

    def load(self, load_path):
        U.load_state(load_path, sess=tf.get_default_session())


def argsparser():
    parser = argparse.ArgumentParser("Obtain attack: use TRPO to train an attacker")
    parser.add_argument('--env_id', help='environment ID', default='attack_and_defense')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--training_iters', help='TRPO training iters', type=int, default=30)
    parser.add_argument('--expert_path', type=str, default='./expert_data/1/0/trajs_neutral.npz')
    parser.add_argument('--reward_giver_path', type=str, default='./expert_data/1/0/reward_giver')
    parser.add_argument('--results_path', type=str, default='./results_de/1/0/')
    parser.add_argument('--nat', help='AS', type=int, default=1)
    parser.add_argument('--ns', help='NS', type=int, default=10)

    return parser.parse_args()


def load_policy(env, policy_fn, path):
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)
    act_params = {
        'name': "pi",
        'ob_space': ob_space,
        'ac_space': ac_space,
    }
    from baselines.common.act_wrapper import ActWrapper

    policy = ActWrapper(pi, act_params)
    policy.load(path, policy_fn)
    return policy


class GenerateExpert(object):
    def __init__(self, args):
        self.args = args

    def train_expert(self):
        print('Training attacker on ', self.args.env_id, ' ...')

        # Create env
        if self.args.env_id == 'attack_and_defense':
            env = generate_env(self.args.env_id, na=args.nat, nt=args.nat + args.ns)\
                .generate(reward_path=self.args.reward_giver_path, neutral_data=self.args.expert_path)
        elif self.args.env_id == 'attack_and_defense_online':
            env = generate_env(self.args.env_id, na=args.nat, nt=args.nat + args.ns)\
                .generate(reward_path=self.args.reward_giver_path, neutral_data=self.args.expert_path, seed=args.seed)
        else:
            env = generate_env(self.args.env_id, na=args.nat, nt=args.nat + args.ns).generate()

        if args.nat > 1:
            def policy(name, ob_space, ac_space, index=None):
                return MlpPolicy_Multi_Mean_Embedding(name=name, ob_space=ob_space, ac_space=ac_space,
                                                      hid_size=[256, 256], feat_size=[[], []], index=index)
        else:
            def policy(name, ob_space, ac_space, index=None):
                return MlpPolicy_No_Mean_Emmbedding(name=name, ob_space=ob_space, ac_space=ac_space,
                                                    hid_size=[256, 256], feat_size=[256], index=index)

        self.expert_policy, info = trainer.learn(env=env, policy_fn=policy, timesteps_per_batch=2500,
                                                 max_iters=self.args.training_iters, max_kl=0.01,
                                                 cg_iters=10, cg_damping=0.1, vf_iters=5, vf_stepsize=1e-3,
                                                 lam=0.98, gamma=0.995)
        print('Saving expert model...')
        self.expert_policy.save(self.args.results_path + 'policy')
        with open(self.args.results_path + 'training_results.pickle', 'wb') as handle:
            pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saving expert data...')
        PolicyEvaluator(env, self.expert_policy, self.args.results_path + 'policy_results', mode='act')
        '''
        self.expert_policy = load_policy(env, policy, self.args.results_path + 'policy')
        PolicyEvaluator(env, self.expert_policy, self.args.results_path + 'policy_results', mode='act')
        '''


if __name__ == "__main__":
    args = argsparser()
    '''
    # For debugging purposes
    import os
    args.env_id = 'attack_and_defense_online'
    args.seed = 0
    args.nat = 5
    args.expert_path = os.path.join(os.getcwd(), 'expert_data_online', str(args.nat), str(args.seed), 'trajs_neutral.npz')
    args.reward_giver_path = os.path.join(os.getcwd(), 'expert_data_online', str(args.nat), str(args.seed), 'reward_giver')
    args.training_iters = 2
    args.results_path = os.path.join(os.getcwd(), 'results_de_online', str(args.nat), str(args.seed))
    args.ns = 10
    '''
    expert = GenerateExpert(args)
    expert.train_expert()

