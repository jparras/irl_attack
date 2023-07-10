import numpy as np
from baselines.gail.dataset.mujoco_dset import Mujoco_Dset
from baselines.gail.adversary import TransitionClassifier
from generate_env import generate_env
import os
import scipy.stats as st
import platform


def obtain_reward_vector(dataset, reward_giver, n_agents, mode):
    if mode == 'samples':
        if n_agents == 0:
            # Obtain the rewards for each state - action pair
            num_transitions = dataset.obs.shape[0]
            rws = np.zeros(num_transitions)
            for i in range(num_transitions):
                rws[i] = reward_giver.get_reward(dataset.obs[i], dataset.acs[i])
        else:
            assert n_agents == len(dataset.obs)
            num_transitions = dataset.obs[0].shape[0]
            rws = np.zeros((n_agents, num_transitions))
            for ag in range(n_agents):
                for i in range(num_transitions):
                    rws[ag, i] = reward_giver.get_reward(dataset.obs[ag][i], dataset.acs[ag][i])
        return rws
    else:
        raise RuntimeError('Defense mode not recognized')


'''
elif mode == 'trajs':
    if n_agents == 0:
        raise NotImplementedError
    else:
        assert n_agents == len(dataset.obs)
        num_transitions = dataset.obs[0].shape[0]
        assert sum(
            [len(dataset.done[0][0][i]) for i in range(len(dataset.done[0][0]))]) == num_transitions

        num_trajs_per_agent = len(dataset.done[0][0])

        rws = [0 for _ in range(n_agents)]
        for ag in range(n_agents):
            i = 0  # Transition index of each agent
            itx = 0  # Index of trajectory for each agent
            tr_reward = []  # To store the reward for the given trace length
            while itx < num_trajs_per_agent:
                tr_aux = 0  # To store the partial reward
                tr_idx = 0  # To store the time index in the partial trajectory
                j = 0  # Index of sample in current trajectory!
                while not dataset.done[0][ag][itx][j]:
                    tr_aux += reward_giver.get_reward(dataset.obs[ag][i], dataset.acs[ag][i]) * df ** tr_idx
                    i += 1
                    j += 1
                    tr_idx += 1
                    if trace_length > 0 and j % trace_length == 0:
                        tr_reward.append(tr_aux)
                        tr_aux = 0
                        tr_idx = 0
                if trace_length < 0:  # If trace length is negative, use all the trace!
                    tr_reward.append(tr_aux)
                itx += 1
            rws[ag] = np.squeeze(np.array(tr_reward))
        rws = np.array(rws)
    '''


class defense_mechanism():
    def __init__(self, def_mech, def_model_path, expert_path, env, def_alpha=0.05, hidden_size=256, entcoeff=1e-3,
                 n_agents=0, df=0.995, trace_length=-1):
        self.def_mech = def_mech
        self.online = False  # Offline defense mechanism
        if self.def_mech == 'reward':
            self.def_alpha = def_alpha
            dataset_neutral = Mujoco_Dset(expert_path=expert_path, traj_limitation=-1, n_agents=n_agents)
            self.def_model = TransitionClassifier(env.observation_space_def, env.action_space, hidden_size,
                                                  entcoeff=entcoeff)
            self.def_model.load_model(os.path.dirname(def_model_path), def_model_path)  # Load the right weights

            rew_data = obtain_reward_vector(dataset_neutral, self.def_model, n_agents, 'samples').flatten()
            #rew_data = np.squeeze(self.def_model.get_reward(dataset_neutral.obs, dataset_neutral.acs))
            sorted_data = np.sort(rew_data)
            cdf = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
            self.def_threshold = sorted_data[np.where(cdf >= self.def_alpha)[0][0] - 1]

    def test(self, obs, acs):
        # Obs, acs = num_transitions x dimensions numpy arrays
        result = False  # False: No attacker detected; True: Attacker detected
        rew_data = None
        if self.def_mech == 'reward':
            rew_data = np.squeeze(self.def_model.get_reward(obs, acs))
            # Use a bimomial to fix the threshold
            probs = st.binom.cdf(np.arange(obs.shape[0]), obs.shape[0], self.def_alpha)
            p_threshold = np.where(probs >= 1 - self.def_alpha)[0][0]
            p = np.sum(rew_data <= self.def_threshold)
            if p >= p_threshold:
                result = True  # Detected as attacker
        else:
            result = False
        return result, rew_data


class defense_mechanism_online():
    def __init__(self, def_mech, def_model_path, expert_path, env, def_alpha=0.05, hidden_size=256, entcoeff=1e-3,
                 n_agents=0, nat=0, seed=0):
        self.def_mech = def_mech
        self.def_model_path = def_model_path
        self.expert_path = expert_path
        self.def_threshold = None
        self.nat = nat
        self.nr_ns = n_agents - nat
        self.seed = seed
        self.n_agents = n_agents
        self.online = True  # Online defense mechanism
        self.nr_anchor = 5  # Number of anchor nodes
        self.n_trajs_to_train = 100  # Trajectories needed to train

        if self.def_mech == 'reward':
            self.def_alpha = def_alpha
            self.def_model = TransitionClassifier(env.observation_space_def, env.action_space, hidden_size,
                                                  entcoeff=entcoeff)

    def load_def(self):

        self.def_model.load_model(os.path.dirname(self.def_model_path), self.def_model_path)  # Load the right weights
        dataset_neutral = Mujoco_Dset(expert_path=self.expert_path, traj_limitation=-1, n_agents=self.nr_anchor)
        rew_data = obtain_reward_vector(dataset_neutral, self.def_model, self.nr_anchor, 'samples').flatten()
        # rew_data = np.squeeze(self.def_model.get_reward(dataset_neutral.obs, dataset_neutral.acs))
        sorted_data = np.sort(rew_data)
        cdf = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        self.def_threshold = sorted_data[np.where(cdf >= self.def_alpha)[0][0] - 1]

    def train_gail(self):  # Calls GAIL on a previously saved set of neutral data
        # Train the reward function using GAIL
        ga_rw = 'generate_gail_classifier.py --env_id=' + 'attack_for_gail' + \
                ' --seed=' + str(self.seed) + \
                ' --expert_path=' + str(self.expert_path) + \
                ' --reward_giver_path=' + str(self.def_model_path) + \
                ' --g_step=' + str(3) + \
                ' --d_step=' + str(1) + \
                ' --num_iters=' + str(10) + \
                ' --nat=' + str(0) + \
                ' --ns=' + str(self.nr_anchor)
        if platform.system() == 'Windows':
            print("Running on Windows")
            _ = os.system('python ' + ga_rw)  # Windows order
        else:
            print("Running on Linux")
            _ = os.system('python3 ' + ga_rw)  # Linux order

        self.load_def()


    def test(self, obs, acs):
        # Obs, acs = num_transitions x dimensions numpy arrays
        result = False  # False: No attacker detected; True: Attacker detected
        rew_data = None
        if self.def_mech == 'reward' and self.def_threshold is not None:
            rew_data = np.squeeze(self.def_model.get_reward(obs, acs))
            # Use a bimomial to fix the threshold
            probs = st.binom.cdf(np.arange(obs.shape[0]), obs.shape[0], self.def_alpha)
            p_threshold = np.where(probs >= 1 - self.def_alpha)[0][0]
            p = np.sum(rew_data <= self.def_threshold)
            if p >= p_threshold:
                result = True  # Detected as attacker
        else:
            result = False
        return result, rew_data


if __name__ == '__main__':
    env = generate_env('attack').generate()
    dm = defense_mechanism('reward', './expert_data/0/reward_giver', './expert_data/0/trajs_neutral.npz', env)
    dataset_neutral = Mujoco_Dset(expert_path='./expert_data/0/trajs_neutral.npz', traj_limitation=-1)
    # Build a dataset of obs - acs
    idx = np.arange(dataset_neutral.obs.shape[0])
    np.random.shuffle(idx)
    tr_per_batch = 20
    results = []
    n_of_tests = int(np.ceil(dataset_neutral.obs.shape[0] / tr_per_batch))
    for i in range(n_of_tests):
        results.append(dm.test(dataset_neutral.obs[idx[i * tr_per_batch: (i+1) * tr_per_batch]],
                               dataset_neutral.acs[idx[i * tr_per_batch: (i+1) * tr_per_batch]]))
    print('N of tests = ', n_of_tests)
    print('Test proportion: attacker detected = ', np.sum(results)/len(results))