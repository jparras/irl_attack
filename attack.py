import numpy as np
import gym
from gym import spaces
from scipy.stats import ncx2
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import chi2
from gym.utils import seeding
from defense import defense_mechanism
from generate_env import generate_env


def get_distance_matrix(points, world_size=None, torus=False, add_to_diagonal=0):
    distance_matrix = np.vstack([get_distances(points, p, torus=torus, world_size=world_size) for p in points])
    distance_matrix = distance_matrix + np.diag(add_to_diagonal * np.ones(points.shape[0]))
    return distance_matrix


def get_distances(x0, x1, torus=False, world_size=None):
    delta = np.abs(x0 - x1)
    if torus:
        delta = np.where(delta > world_size / 2, delta - world_size, delta)
    dist = np.sqrt((delta ** 2).sum(axis=-1))
    return dist

class AttackEnv(gym.Env):

    def __init__(self, nr_agents=5,  # Agents in the swarm
                 nr_agents_total=10,  # Total number of agents (i.e., AS + NS)
                 obs_mode='sum_obs', # Observation mode (for attackers)
                 obs_returned='attack',  # Selects the by default observation returned
                 attack_mode='phy', # PHY or MAC attack
                 obs_radius=5000, # Observation radius of the sensors (if all sensors have comm: make it large enough)
                 world_size=1000, # Size of the world
                 lambda_phy=0.4,  # PHY attack threshold (-1 for random changing)
                 phy_agent_memory=5, # Length of PHY agent memory (obs stored)
                 K=5,  # MAC attack parameter
                 L=1000,  # MAC defense parameter
                 lambda_mac=0.4, # MAC attack threshold (-1 for random changing)
                 timesteps_limit=256,  # Max number of timesteps for the attack
                 sum_rwd=True, # If true, all ASs have as reward the sum of the rewards of all ASs
                 df=None,  # To obtain discounted reward
                 #nr_neutral_agents = 0,  # Number of neutral agents to return: only used if nr_agents = 0
                 def_mech=None,  # Defense mechanism
                 def_trace_length=5  # Trace length for defense mechanism
                 ):

        self.nr_agents = nr_agents
        self.nr_neutral_agents = 0
        #if self.nr_agents == 0:
        #    self.nr_neutral_agents = nr_neutral_agents  # In this case, the returned values are for a neutral agent
        self.nr_agents_total = nr_agents_total
        self.nr_normal_sensors = int(self.nr_agents_total - self.nr_agents)
        self.world_size = world_size
        self.obs_mode = obs_mode
        self.obs_radius = obs_radius
        self.attack_mode = attack_mode
        self.K = K
        self.phy_agent_memory = phy_agent_memory
        self.timestep = None  # Each timestep is a decision, NOT actual physical time!!
        self.timestep_limit = timesteps_limit  # Max number of timesteps per episode
        self.sum_rwd = sum_rwd  # Flag: whether to return the sum of rewards in simulate_episode method
        self.disc_factor = df  # Discount factor
        self.def_mech = def_mech # Defense mechanism used
        self.def_trace_length = def_trace_length
        self.obs_returned = obs_returned

        if self.attack_mode == 'phy':
            self.at_obs_size = self.phy_agent_memory + 3  # Append SNR and detected flag and index of agent
            self.action_size = 1  # Continuous actions
        elif self.attack_mode == 'mac':
            self.at_obs_size = self.K * 2 + 3  # Append current backoff and detected flag and index of agent
            self.action_size = 2  # Discrete actions: whether to transmit or not
        else:
            raise RuntimeError("Attack mode not recognized")

        if self.obs_mode == 'sum_obs':
            self.dim_rec_o = (self.nr_agents_total - 1, self.at_obs_size + 2)  # Add 2 values for indexation
            self.dim_mean_embs = self.dim_rec_o
            self.dim_flat_o = self.at_obs_size
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
        elif self.obs_mode == 'sum_obs_multi':
            self.dim_flat_o = self.at_obs_size
            self.dim_at_obs = (self.nr_agents, self.at_obs_size + 2)  # Add 2 values for indexation
            self.dim_nr_obs = (self.nr_normal_sensors, self.at_obs_size + 2)  # Add 2 values observation for indexation
            self.dim_rec_o = self.dim_at_obs
            self.dim_mean_embs = (self.dim_at_obs,) + (self.dim_nr_obs,)
            self._dim_o = np.prod(self.dim_at_obs) + np.prod(self.dim_nr_obs) + self.dim_flat_o
        else:
            raise NotImplementedError

        self.banned = None  # To store banned sensors from the network
        self.banned_phy = None  # To store banned sensors from the network due to phy def mechanism
        self.banned_mac = None  # To store banned sensors from the network due to mac def mechanism
        self.agent_pos = None
        self.distance_matrix = None
        self.rewards_hist = None
        self.aux = [[] for _ in range(self.nr_agents_total)]

        # Initialize simulation values
        if self.attack_mode == 'mac':
            self.lambda_mac = lambda_mac  # Detection threshold to detect a malicious user in MAC attack
            self.mac_rep = None  # MAC reputation
            self.mac_rep_time = None
            self.L = L  # MAC defense parameter
            self.t_mac_max = 5e5  # us of the simulation
            self.Rb = 1  # Mbps
            self.MAC_header = 272 / self.Rb  # us
            self.PHY_header = 128 / self.Rb  # us
            self.H = self.MAC_header + self.PHY_header
            self.ACK = (112 + self.PHY_header) / self.Rb  # us
            self.RTS = (160 + self.PHY_header)  # us
            self.CTS = (272 + self.PHY_header)  # us
            self.d = 1  # us
            self.SIFS = 28  # us
            self.DIFS = 128  # us
            self.t_mac = None # t, in us
            self.fr_size = 4096  # Fixed
            self.CWmin = 32
            self.CWmax = 1024
            self.tx_per_test = 5  # Controls how often the defense mechanism is invoqued
            self.success_counter = None  # successful transmisions counter
            self.collision_counter = None  # collision counters
            self.t_counter = None
            self.current_backoffs = None
            self.hist_backoffs = None
            self.event_list = None
            self.state_action_list = None
            self.tmt = None  # Number of MAC txs per sensor
            self.tmc = None  # Number of mac collisions
            self.cw_vector = None  # Current backoff state
            self.t_tx = None
            self.t_col = None

        if self.attack_mode == 'phy':
            self.lambda_phy = lambda_phy  # Detection threshold to detect a malicious user in PHY
            self.phy_rep = None  # PHY reputation
            self.distances = None
            self.gamma = None  # To store SNR of each sensor
            self.gamma_norm = None  # SNR normalized
            self.energy_norm = None  # Normalization value for the energy!!
            self.energy_hist = None
            self.fc_decisions = None  # to store FC decisions
            # Transmission
            self.Pl0 = 35  # Path loss at 1 meter
            self.path_loss = 3  # Path loss exponent
            self.Ptx = 23  # In dBm (200 mW)
            self.m = 5
            self.npw = -110  # dBm
            self.DC = 0.2  # %Duty cycle
            # Defense parameters
            self.eta = 1
            self.zeta = 1.6
            self.r = None
            self.s = None
            self.reports_per_test = 1  # Controls how often the defense mechanism is invoqued
            self.channel_state = None

    @property
    def observation_space(self):
        ob_space = spaces.Box(low=0., high=1., shape=(self._dim_o,), dtype=np.float32)
        ob_space.dim_local_o = self.at_obs_size
        ob_space.dim_flat_o = self.dim_flat_o
        ob_space.dim_rec_o = self.dim_rec_o
        ob_space.dim_mean_embs = self.dim_mean_embs
        return ob_space

    @property
    def observation_space_def(self):
        ob_space = spaces.Box(low=0., high=1., shape=(self.K * 2 + 1,), dtype=np.float32)
        return ob_space

    @property
    def action_space(self):
        if self.attack_mode == 'mac':
            return spaces.Discrete(self.action_size)
        elif self.attack_mode == 'phy':
            return spaces.Box(low=0., high=1., shape=(self.action_size,), dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def reset(self): # Method to reset the environment
        if self.attack_mode == 'mac' and self.lambda_mac < 0:
            self.lambda_mac = np.random.uniform(low=0.2, high=0.8)
            assert 0 <= self.lambda_mac <= 1
        if self.attack_mode == 'phy' and self.lambda_phy < 0:
            self.lambda_phy = np.random.uniform(low=0.2, high=0.8)
            assert 0 <= self.lambda_phy <= 1

        self.timestep = 0
        self.banned = np.zeros(self.nr_agents_total, dtype=bool)  # No sensor banned yet
        self.banned_phy = np.zeros(self.nr_agents_total, dtype=bool)  # No sensor banned yet
        self.banned_mac = np.zeros(self.nr_agents_total, dtype=bool)  # No sensor banned yet
        # Initialize the position of each agent
        angle = np.random.uniform(low=0, high=2 * np.pi, size=self.nr_agents_total)
        dist = np.random.uniform(low=0.8 * self.world_size, high=self.world_size, size=self.nr_agents_total)
        self.agent_pos = np.zeros((self.nr_agents_total, 2))
        self.agent_pos[:, 0] = dist * np.cos(angle)
        self.agent_pos[:, 1] = dist * np.sin(angle)
        self.distance_matrix = get_distance_matrix(self.agent_pos, torus=False, world_size=self.world_size,
                                                     add_to_diagonal=-1)
        self.rewards_hist = []
        if self.attack_mode == 'phy':
            # Obtain SNR in each sensor
            self.distances = np.linalg.norm(self.agent_pos, axis=1)  # Distances to the origin (i.e., primary is here)
            loss = self.Pl0 + 10 * self.path_loss * np.log10(self.distances)
            E_r_m = self.Ptx - loss  # Energy in each sensor
            gamma_dB = E_r_m - self.npw  # Average SNR in each sensor
            self.gamma = np.power(10, gamma_dB / 10)  # SNR in linear units!!
            self.gamma_norm = self.gamma / np.amax(self.gamma)
            # Obtain attack parameters for each attacking sensor!!
            if self.nr_agents > 0:
                self.energy_norm = np.amax(ncx2.ppf(0.9, 2 * self.m, 2 * self.gamma[0:self.nr_agents]))
            else:
                # Normalization value: taken from all the sensors - only used for normalization purposes
                self.energy_norm = np.amax(ncx2.ppf(0.9, 2 * self.m, 2 * self.gamma))
            # Defense parameters
            self.r = np.zeros(self.nr_agents_total)
            self.s = np.zeros(self.nr_agents_total)
            self.phy_rep = np.zeros([self.nr_agents_total, self.timestep_limit])  # To store trust
            # History values
            self.energy_hist = np.zeros([self.nr_agents_total, self.timestep_limit])  # To store energy history
            self.fc_decisions = np.zeros(2)  # To store decisions of the FC: 0 and 1

        if self.attack_mode == 'mac':
            self.mac_rep = [[] for _ in range(self.nr_agents_total)]
            self.mac_rep_time = [[] for _ in range(self.nr_agents_total)]
            self.event_list = [[] for _ in range(self.nr_agents_total)]
            self.state_action_list = [[] for _ in range(self.nr_agents_total)]
            self.t_mac = 0  # Start time (physical time)
            self.success_counter = 0
            self.collision_counter = 0
            self.t_counter = np.zeros(2)  # In 0, we put time used in tx, in [1], time not used in tx
            self.current_backoffs = np.zeros(self.nr_agents_total)
            self.hist_backoffs = [[] for _ in range(self.nr_agents_total)]
            self.tmt = np.zeros(self.nr_agents_total)  # Total MAC txs for each sensor
            self.tmc = np.zeros(self.nr_agents_total)  # Total MAC cols for each sensor
            self.cw_vector = self.CWmin * np.ones(self.nr_agents_total)
            # Init normal sensors backoffs!!
            self.current_backoffs[self.nr_agents:] = np.random.randint(1, high=self.CWmin, size=self.nr_normal_sensors)
            self.t_tx = int(round(self.H + self.fr_size / self.Rb + self.SIFS + self.d + self.ACK + self.DIFS + self.d))
            self.t_col = int(self.H + self.DIFS + self.d + self.fr_size / self.Rb)
            for s in range(self.nr_normal_sensors):
                idx = int(self.nr_agents + s)
                self.hist_backoffs[idx].append(self.current_backoffs[idx])

        return self.get_obs()

    def get_obs(self):
        if self.obs_returned == 'attack':
            return self.get_att_obs()
        elif self.obs_returned == 'def':
            return self.get_def_obs()
        elif self.obs_returned == 'def_single':
            return self.get_def_obs()[0]
        else:
            raise NotImplementedError


    def get_att_obs(self):
        obs = []
        local_obs = self.get_local_observations()
        if self.nr_agents > 0:
            for s in range(self.nr_agents):
                ob = self.build_observation(self.distance_matrix[s, :], local_obs, s)
                obs.append(ob)
        return obs

    def build_observation(self, dm, states, s):

        if self.obs_mode == 'sum_obs':
            in_range = (dm < self.obs_radius) & (0 < dm)
            nr_neighbors = np.sum(in_range)
            local_obs = states[s, :]
            assert len(local_obs) == self.dim_flat_o
            # local_obs[-1] = nr_neighbors / (self.n_agents - 1)

            sum_obs = np.zeros(self.dim_rec_o)
            sum_obs[0:nr_neighbors, 0:self.at_obs_size] = states[in_range, :]
            sum_obs[0:nr_neighbors, -2] = 1
            sum_obs[0:self.nr_agents_total - 1, -1] = 1

            obs = np.hstack([sum_obs.flatten(), local_obs])

        elif self.obs_mode == 'sum_obs_multi':
            in_range = (dm < self.obs_radius) & (0 < dm)
            at_idx = np.hstack(
                [np.ones(self.nr_agents, dtype=bool).flatten(), np.zeros(self.nr_normal_sensors, dtype=bool).flatten()])
            ns_idx = np.hstack(
                [np.zeros(self.nr_agents, dtype=bool).flatten(), np.ones(self.nr_normal_sensors, dtype=bool).flatten()])
            nr_at_neighbors = np.sum(in_range[0:self.nr_agents])
            nr_ns_neighbors = np.sum(in_range[self.nr_agents:])
            local_obs = states[s, :]
            assert len(local_obs) == self.dim_flat_o

            sum_obs_at = np.zeros(self.dim_at_obs)
            sum_obs_at[0:nr_at_neighbors, 0:self.at_obs_size] = states[in_range * at_idx, :]
            sum_obs_at[0:nr_at_neighbors, -2] = 1
            sum_obs_at[0:self.nr_agents, -1] = 1

            sum_obs_ns = np.zeros(self.dim_nr_obs)
            sum_obs_ns[0:nr_ns_neighbors, 0:self.at_obs_size] = states[in_range * ns_idx, :]
            sum_obs_ns[0:nr_ns_neighbors, -2] = 1
            sum_obs_ns[0:self.nr_normal_sensors, -1] = 1

            obs = np.hstack([sum_obs_at.flatten(), sum_obs_ns.flatten(), local_obs])

        else:
            raise NotImplementedError

        return obs

    def get_local_observations(self):
        lobs = np.zeros((self.nr_agents_total, self.at_obs_size))  # Each agents own state
        # Obtain the local obs of each agent
        for s in range(self.nr_agents_total):
            # Obtain index of each agent
            if s < self.nr_agents:  # Attacker
                index = (s + 1) / self.nr_agents  # Strictly larger than 0
            else:
                index = 0

            if self.attack_mode == 'phy':
                obs = self.get_phy_obs(s)
                aux = np.append(obs, np.array(self.banned[s].astype(int))).flatten()  # Append detected flag
                aux = np.append(aux, np.array(self.gamma_norm[s])).flatten()  # Append SNR normalized
                lobs[s, :] = np.append(aux, np.array(index)).flatten()
            elif self.attack_mode == 'mac':
                obs = self.get_mac_obs(s)
                aux = np.append(obs, np.array(self.banned[s].astype(int))).flatten()  # Append detected flag
                aux = np.append(aux, np.array(np.clip(self.current_backoffs[s]/self.CWmin, 0, 1))).flatten()
                lobs[s, :] = np.append(aux, np.array(index)).flatten()
            else:
                raise NotImplementedError
        return lobs

    def get_phy_obs(self, s):
        if self.timestep >= self.phy_agent_memory:
            phy_obs = self.energy_hist[s, self.timestep - self.phy_agent_memory: self.timestep]
        else:
            phy_obs = -np.ones(self.phy_agent_memory)
            if self.timestep > 0:
                phy_obs[-self.timestep:] = self.energy_hist[s, 0: self.timestep]

        return np.clip(phy_obs / self.energy_norm, 0, 1)  # Normalized and clip energy values

    def get_mac_obs(self, s):
        bc = self.event_list[s]
        mac_obs = np.zeros(2 * self.K)
        for id in range(min([len(bc), self.K])):
            mac_obs[-2 * id - 1] = bc[id - 1][2]  # Flag value
            mac_obs[-2 * id - 2] = (self.timestep - bc[id - 1][0]) / self.timestep_limit  # Differential and normalized
        return mac_obs

    def get_def_obs(self):
        obs = []
        local_obs = self.get_local_observations_def()
        for s in range(self.nr_agents_total):
            #ob = self.build_observation_def(self.distance_matrix[s, :], local_obs, s)
            #ob = np.zeros(self.observation_space_def.shape[0])
            #ob[0: len(local_obs[s])] = local_obs[s]
            obs.append(local_obs[s])
        return obs

    '''
    def build_observation_def(self, dm, states, s):

        in_range = (0 < dm)  # The defense mechanism observes all sensors!! (Star topology)
        nr_neighbors = np.sum(in_range)
        local_obs = states[s, :]
        # assert len(local_obs) == self.dim_flat_o
        # local_obs[-1] = nr_neighbors / (self.n_agents - 1)

        sum_obs = np.zeros(self.dim_rec_o)
        sum_obs[0:nr_neighbors, 0:self.at_obs_size] = states[in_range, :]
        sum_obs[0:nr_neighbors, -2] = 1
        sum_obs[0:self.nr_agents_total - 1, -1] = 1

        obs = np.hstack([sum_obs.flatten(), local_obs])

        return obs
    '''

    def get_local_observations_def(self):
        lobs = np.zeros((self.nr_agents_total, self.observation_space_def.shape[0]))  # Each agents own state
        # Obtain the local obs of each agent
        for s in range(self.nr_agents_total):
            # Obtain index of each agent
            if self.attack_mode == 'phy':
                raise NotImplementedError
            elif self.attack_mode == 'mac':
                obs = self.get_mac_obs_def(s)
                lobs[s, :] = np.append(obs, np.array(self.banned[s].astype(int))).flatten()  # Append detected flag
            else:
                raise NotImplementedError
        return lobs


    def get_mac_obs_def(self, s):
        bc = self.event_list[s]
        mac_obs = np.zeros(2 * self.K)
        for id in range(min([len(bc), self.K])):
            mac_obs[-2 * id - 1] = bc[id - 1][2]  # Flag value
            mac_obs[-2 * id - 2] = (self.timestep - bc[id - 1][0]) / self.timestep_limit  # Differential and normalized
        return mac_obs

    def step_phy(self, actions):

        reward = np.zeros(self.nr_agents)
        done = False
        # Generate energies
        tx = np.random.binomial(1, self.DC)  # Flag: whether primary transmits or not
        if tx == 1:
            # Generate energies if primary transmits
            E_v = np.random.noncentral_chisquare(2 * self.m * np.ones(self.nr_agents_total), self.gamma)
        else:
            # Generate energies if primary does not transmit
            E_v = np.random.chisquare(2 * self.m, self.nr_agents_total)

        if self.nr_agents > 0:
            # ATTACK: modify energies in attacking sensors not banned
            E_va = actions * self.energy_norm + np.finfo(np.float32).eps  # Energy must be strictly higher than 0
            for s in range(self.nr_agents):
                if not self.banned[s]:
                    E_v[s] = E_va[s]
        self.energy_hist[:, self.timestep] = E_v

        # DEFENSE
        if self.timestep % self.reports_per_test == 0:
            # Perform test #1
            t1 = np.ones(self.nr_agents_total)  # To store results of test 1
            np_value = ncx2.pdf(E_v, 2 * self.m, 2 * self.gamma) / chi2.pdf(E_v, 2 * self.m)  # Neyman Pearson values!
            for s in range(self.nr_agents_total):
                mask = np.ones(self.nr_agents_total, dtype=bool)
                mask[s] = False
                t1[s] = np.prod(np_value[mask])
            # Decide in the FC: use majority rule
            if np.sum(t1[np.logical_not(self.banned)] > self.eta) > \
                    np.sum(t1[np.logical_not(self.banned)] <= self.eta):  # Primary detected
                self.fc_decisions[1] += 1
                self.channel_state = 1
                reward[:] += 1  # Reward for causing the FC to believe primary is present
            else:  # No primary detected by FC
                reward[:] += 0  # Reward for causing the FC to believe primary is absent
                self.fc_decisions[0] += 1  # OK
                self.channel_state = 0

            # Perform test 2
            valid_index = np.logical_and(t1 < self.eta, np.logical_not(self.banned))  # No primary detected
            t2 = np.zeros(self.nr_agents_total)
            t2[valid_index] = chi2.pdf(E_v[valid_index] - 2 * self.gamma[valid_index], 2 * self.m) / \
                              chi2.pdf(E_v[valid_index], 2 * self.m)

            # Update r and s
            node_lies = np.logical_not(self.banned)
            node_lies = np.logical_and(node_lies, np.logical_and(t1 < self.eta, t2 > self.zeta))
            self.s[node_lies] += 1
            self.r[np.logical_not(node_lies)] += 1
            # Update trust
            self.phy_rep[:, self.timestep] = (self.r + 1) / (self.r + self.s + 1)
            # Detect ASs
            new_banned_sensors = np.logical_and(np.logical_not(self.banned),
                                                self.phy_rep[:, self.timestep] < self.lambda_phy)
            self.banned[new_banned_sensors] = True
            self.banned_phy[new_banned_sensors] = True

        if self.timestep >= self.timestep_limit - 1:
            done = True

        return reward, done

    def step_mac(self, actions):

        reward = np.zeros(self.nr_agents) if self.nr_agents > 0 else np.zeros(self.nr_agents_total)
        done = False
        # Simulate a transmission
        # Update backoff of the attackers (if any)
        if self.nr_agents > 0:  # Attack
            actions = np.random.binomial(1, actions)  # Actions realizations
            for s in range(self.nr_agents):
                if actions[s] == 0:  # No attack
                    # The backoff increase just assures that the backoff is positive and hence, there is no tx.
                    # Also saves the backoff value for hist_backoffs
                    self.current_backoffs[s] += 1
                elif self.banned[s]:
                    self.current_backoffs[s] = -2  # Banned sensor does not transmit!
                else:  # Attack made by a non-detected sensor
                    self.hist_backoffs[s].append(self.current_backoffs[s])
                    self.current_backoffs[s] = 0  # To indicate that there is transmission
        self.current_backoffs[self.nr_agents:] -= 1  # Decrease backoff in normal sensors
        self.current_backoffs[self.banned] = -2  # Banned sensors do not transmit!
        self.t_counter[1] += 1  # Time not used in tx!!
        self.t_mac = sum(self.t_counter)  # Update time of MAC simulation
        actions_list = np.array(self.current_backoffs == 0).astype(int)
        if np.sum(self.current_backoffs == 0) == 1:  # Exactly one sensor starts to transmit
            self.t_counter[0] += self.t_tx  # Time used in tx!!
            self.t_mac = sum(self.t_counter)  # Update time of MAC simulation
            self.success_counter += 1
            s = np.where(self.current_backoffs == 0)[0][0]  # Sensor that transmits
            self.tmt[s] += 1  # Add another successful transmission to sensor s
            self.event_list[s].append([self.timestep, self.t_mac, 1])  # 1 is for tx
            if s < self.nr_agents:  # Attacker sensor transmitting
                reward[s] += 0  # Update reward: successful transmission
            else:  # Normal sensor transmitting
                reward[:] += -1  # Normal sensor transmitted
                self.cw_vector[s] = self.CWmin  # After tx use the min window
                self.current_backoffs[s] = np.random.randint(1, high=self.cw_vector[s])
                self.hist_backoffs[s].append(self.current_backoffs[s])

        elif np.sum(self.current_backoffs == 0) > 1:  # Collision
            self.t_counter[1] += self.t_col  # Time not used in tx!!
            self.t_mac = sum(self.t_counter)  # Update time of MAC simulation
            self.collision_counter += 1
            node_tx = np.where(self.current_backoffs == 0)[0]  # Sensors that transmits
            for s in node_tx:
                self.tmc[s] += 1
                self.event_list[s].append([self.timestep, self.t_mac, 0])  # 0 is for col
                if s < self.nr_agents:  # Attacking sensor
                    reward[s] += 0  # Update reward: collision of attacker
                else:  # Normal sensor
                    reward[:] += 0  # Update reward: collision of normal sensor
                    self.cw_vector[s] *= 2  # Duplicate backoff window
                    if self.cw_vector[s] > self.CWmax:
                        self.cw_vector[s] = self.CWmax
                    self.current_backoffs[s] = np.random.randint(1, high=self.cw_vector[s])
                    self.hist_backoffs[s].append(self.current_backoffs[s])

        # Defense test
        if self.timestep % self.tx_per_test == 0 and self.timestep > 0:
            # Obtain pc with +1 in the denominator, to avoid pc=1 (this causes the defense mech to malfunction)
            pc = self.collision_counter / (self.collision_counter + self.success_counter + 1)
            cdf_x_values = range(self.CWmax + 1)
            probs = np.array([1 - pc, pc * (1 - pc), (pc ** 2) * (1 - pc), (pc ** 3) * (1 - pc),
                              (pc ** 4) * (1 - pc)])  # Assume a limit in nc
            probs = np.append(probs, 1 - np.sum(probs))  # Make it a distribution
            probs = np.round(self.L * probs)
            Lact = int(sum(probs))  # Actual number of samples to take

            cw_vals = [32, 64, 128, 256, 512, 1024]
            y = []  # To store theoretical distribution
            for i in range(len(cw_vals)):
                if probs[i] > 0:
                    idx = np.random.choice(i + 1, size=int(probs[i]))
                    for udist in range(i + 1):
                        y.extend(list(np.random.randint(low=0, high=cw_vals[udist], size=np.sum(idx == udist))))
            y = np.array(y).flatten()

            ecdf = sm.distributions.ECDF(y)
            F0 = ecdf(cdf_x_values)
            # Obtain empirical CDF for each sensor
            for s in range(self.nr_agents_total):
                if not self.banned[s]:
                    x = self.hist_backoffs[s]
                    if len(x) >= self.K:  # I have at least K observations
                        x = np.array(x[-self.K :])  # Take only last K values
                        # Obtain empirical CDF F1
                        ecdf = sm.distributions.ECDF(x)
                        F1 = ecdf(cdf_x_values)
                        # Test: modified CvM test
                        sum1 = 0
                        for i in range(self.K):
                            sum1 = sum1 + np.sign(F0[int(x[i])] - F1[int(x[i])]) * (F0[int(x[i])] - F1[int(x[i])]) ** 2
                        sum2 = 0
                        for i in range(Lact):
                            sum2 = sum2 + np.sign(F0[int(y[i])] - F1[int(y[i])]) * (F0[int(y[i])] - F1[int(y[i])]) ** 2
                        theta = (sum1 + sum2) * self.K * self.L / ((self.K + self.L) ** 2)
                        D = min([theta, 0])
                        self.mac_rep[s].append(np.exp(-D ** 2))
                        self.mac_rep_time[s].append(self.t_mac)
                        if self.mac_rep[s][-1] < self.lambda_mac:
                            self.event_list[s].append([self.timestep, self.t_mac, -1])  # -1 is for detection
                            self.banned[s] = True  # Banned from network!
                            self.banned_mac[s] = True
                            self.current_backoffs[s] = -2
        self.t_mac = sum(self.t_counter)
        if self.t_mac >= self.t_mac_max:  # finish only when simulation time is satisfied
            done = True

        return reward, done, actions_list


    def step(self, actions):

        # Simulation differs if attack is PHY or MAC!!
        if isinstance(actions, np.int64):
            actions = np.array([actions])

        if self.nr_agents > 0:
            assert len(actions) == self.nr_agents
            assert np.amin(actions) >= 0
            assert np.amax(actions) <= 1
        # Take a step
        ac_list = None  # To store normal sensor actions
        if self.attack_mode == 'phy':
            reward, done = self.step_phy(actions)
            self.timestep += 1  # Increase timestep
        elif self.attack_mode == 'mac':
            reward, done, ac_list = self.step_mac(actions)  # ac_list includes the actions of normal sensors!!
            self.timestep += 1  # Increase timestep
            # Obtain final reward if all agents have already been discovered (in MAC attack, we run for a fixed time!)
            if np.sum(self.banned[0:self.nr_agents]) == self.nr_agents \
                    and self.nr_agents > 0 and self.obs_returned == 'attack':
                while not done:
                    actions = np.zeros_like(actions) if self.nr_agents > 0 else []
                    r_mac, done, _ = self.step_mac(actions)
                    self.timestep += 1
                    reward += r_mac
        else:
            raise RuntimeError("Unrecognized step order to the environment")


        # Update state_action list
        def_obs = self.get_def_obs()
        for s in range(self.nr_agents_total):
            self.state_action_list[s].append([def_obs[s], ac_list[s]])

        if self.def_mech is not None:
            #aux = np.zeros(self.nr_agents_total)
            for s in range(self.nr_agents_total):
                if len(self.state_action_list[s]) > 0 and len(self.state_action_list[s]) % self.def_trace_length == 0:
                #if len(self.state_action_list[s]) > self.def_trace_length:
                    vals = self.state_action_list[s][-self.def_trace_length:]
                    #vals = self.state_action_list[s][:]
                    obs = np.array([val[0] for val in vals])
                    acs = np.array([val[1] for val in vals])
                    acs = acs.reshape([acs.size, 1])
                    result, rw = self.def_mech.test(obs, acs)
                    self.aux[s].append(rw)
                    #if result:
                    #    aux[s] = 1
                    if result:
                        self.banned_mac[s] = True
                        self.banned[s] = True
            #print(aux)

        if (np.sum(self.banned[0:self.nr_agents]) == self.nr_agents and self.nr_agents > 0) or \
                np.sum(self.banned) == self.nr_agents_total:
            if np.sum(self.banned[0:self.nr_agents]) == self.nr_agents \
                    and self.nr_agents > 0 and self.obs_returned == 'attack' and self.attack_mode=='mac':
                while not done:
                    actions = np.zeros_like(actions) if self.nr_agents > 0 else []
                    r_mac, done, _ = self.step_mac(actions)
                    self.timestep += 1
                    reward += r_mac
            else:
                done = True  # All agents have been discovered

        if self.timestep >= self.timestep_limit:
            done = True # In the MAC attack, set a maximum number of timesteps: if sensors learn not to transmit, they will cause an error in the F distributions

        self.rewards_hist.append(np.copy(reward))  # Save reward values in memory to obtain info

        if self.sum_rwd:  # Sum of rewards for all agents!!
            aux = np.sum(reward)
            reward[:] = aux  # All agents have the same reward!

        if self.obs_returned == 'def_single':
            reward = reward[0]  # Return a single value!

        if done:
            info = self.get_simulation_info()
        else:
            info = {}
        info['ac_list'] = ac_list
        info['def_obs'] = def_obs
        return self.get_obs(), reward, done, info

    def plot_rep(self, save=False, show=True, title=None):
        if self.attack_mode == 'phy':
            self.phy_plot(save, show, title)
        if self.attack_mode == 'mac':
            self.mac_plot(save, show, title)

    def phy_plot(self, save, show, title):
        t = np.arange(self.timestep_limit)  # Time values
        v = np.zeros_like(t, dtype=bool)  # Plot only times when there was a test
        for n in range(self.timestep_limit):
            if n % self.reports_per_test == 0 and n > 0:
                v[n] = True
        plt.plot(t[v], self.phy_rep[self.nr_agents:, t[v]].T, 'b', alpha=0.5)
        for s in range(self.nr_agents):
            if self.banned[s]:
                plt.plot(t[v], self.phy_rep[s, t[v]].T, 'r')
            else:
                plt.plot(t[v], self.phy_rep[s, t[v]].T, 'g')
        plt.plot(t[v], self.lambda_phy * np.ones_like(t)[v], 'k')
        if save:
            plt.savefig(str(self.nr_agents) + '_' + str(self.nr_agents_total) + '_'
                        + str(np.random.randint(low=0, high=1024)) + '_phy.png')
        if title is None:
            pass
        else:
            plt.title(title)
        if show:
            plt.show()

    def mac_plot(self, save, show, title):
        t_col = []
        t_tx_norm = []
        t_tx_ag = []
        for s in range(self.nr_agents_total):
            for l in range(len(self.event_list[s])):
                if self.event_list[s][l][2] == 0:  # Col
                    t_col.append(self.event_list[s][l][1])
                elif self.event_list[s][l][2] == 1 and s < self.nr_agents:  # Tx, agent
                    t_tx_ag.append(self.event_list[s][l][1])
                elif self.event_list[s][l][2] == 1 and s >= self.nr_agents:  # Tx, normal
                    t_tx_norm.append(self.event_list[s][l][1])
        t_max = max(t_col + t_tx_ag + t_tx_norm)
        for s in reversed(range(self.nr_agents_total)):  # The order is reversed, so that in plot, agents are forward
            if len(self.mac_rep_time[s]) > 0:
                t = np.array([0] + self.mac_rep_time[s])
                t_max = np.amax(t) if np.amax(t) > t_max else t_max
                if s < self.nr_agents and self.banned[s]:
                    plt.plot(t, np.array([1] + self.mac_rep[s]), 'ro-')
                elif s < self.nr_agents and not self.banned[s]:
                    plt.plot(t, np.array([1] + self.mac_rep[s]), 'go-')
                else:
                    plt.plot(t, np.array([1] + self.mac_rep[s]), 'bo-')
        plt.plot([0, t_max], self.lambda_mac * np.ones(2), 'k')
        # Add col and tx
        n = len(t_col) + len(t_tx_norm) + len(t_tx_ag)
        t_col.sort()
        t_tx_norm.sort()
        t_tx_ag.sort()
        plt.plot([0] + t_col + [t_max], np.array([0] + list(range(len(t_col))) + [len(t_col)]) / n, 'm', alpha=0.5)
        plt.plot([0] + t_tx_ag + [t_max], np.array([0] + list(range(len(t_tx_ag))) + [len(t_tx_ag)]) / n, 'g', alpha=0.5)
        plt.plot([0] + t_tx_norm + [t_max], np.array([0] + list(range(len(t_tx_norm))) + [len(t_tx_norm)]) / n, 'b', alpha=0.5)
        if save:
            from matplotlib2tikz import save as tikz_save
            name = str(self.nr_agents) + '_' + str(self.nr_agents_total) + '_' + \
                   str(np.random.randint(low=0, high=1024)) + '_mac'
            tikz_save(name + '.tikz',figureheight='\\figureheight', figurewidth='\\figurewidth')
            plt.savefig(name + '.png')
        if title is None:
            pass
        else:
            plt.title(title)
        if show:
            plt.show()

    def get_simulation_info(self):

        info = {'attackers_caught': np.sum(self.banned[0:self.nr_agents]),
                'attackers_not_caught': self.nr_agents - np.sum(self.banned[0:self.nr_agents]),
                'ns_caught': np.sum(self.banned[self.nr_agents:]),
                'ns_not_caught': self.nr_agents_total - self.nr_agents - np.sum(self.banned[self.nr_agents:]),
                'total_rwd': np.sum(np.array(self.rewards_hist))}
        # Return discounted mean reward!
        df_vector = np.power(self.disc_factor, np.arange(np.array(self.rewards_hist).shape[0]))
        df_matrix = np.array([df_vector for _ in range(np.array(self.rewards_hist).shape[1])]).T
        info['mean_total_rwd'] = np.mean(np.sum(np.array(self.rewards_hist) * df_matrix, axis=0))
        if self.attack_mode == 'phy':
            info['phy_fc_error_rate'] = self.fc_decisions[1] / np.sum(self.fc_decisions)  # Prop of 1 detected

        if self.attack_mode == 'mac':
            info['state_action_list'] = self.state_action_list
            info['total_mac_tx'] = self.success_counter
            info['total_mac_col'] = self.collision_counter
            info['prop_t_tx'] = self.t_counter[0] * 100 / sum(self.t_counter) if sum(self.t_counter)> 0 else 0
            btx_s = self.tmt * self.fr_size
            info['total_bits_tx'] = np.sum(btx_s)
            if np.sum(btx_s) > 0:  # There is transmission
                info['mean_prop_bits_tx_at'] = np.sum(btx_s[0:self.nr_agents]) * 100 / np.sum(btx_s)
                info['mean_prop_bits_tx_no'] = np.sum(btx_s[self.nr_agents:]) * 100 / np.sum(btx_s)
            else:
                info['mean_prop_bits_tx_at'] = 0
                info['mean_prop_bits_tx_no'] = 0
        # Aditional values not used for TRPO, but used in policy evaluation
        info["banned"] = self.banned
        info["banned_phy"] = self.banned_phy
        info["banned_mac"] = self.banned_mac
        info["agent_pos"] = self.agent_pos
        info["distance_matrix"] = self.distance_matrix
        info["rewards_hist"] = self.rewards_hist
        if self.attack_mode == 'phy':
            info['phy_rep'] = self.phy_rep
            info['energy_hist'] = self.energy_hist
            info['energy_norm'] = self.energy_norm
            info['fc_decisions'] = self.fc_decisions

        if self.attack_mode == 'mac':
            info["mac_rep"] = self.mac_rep
            info["mac_rep_time"] = self.mac_rep_time
            info["event_list"] = self.event_list
            info["t_mac"] = self.t_mac
            info["t_counter"] = self.t_counter
            info["hist_backoffs"] = self.hist_backoffs
            info["tmt"] = self.tmt
            info["tmc"] = self.tmc
            info["t_tx"] = self.t_tx
            info["t_col"] = self.t_col
        return info

    def run_baseline(self, policy_type, num_episodes=50):
        import time
        t = time.time()
        total_rewards = np.zeros(num_episodes)
        if policy_type =='random':
            for ep in range(num_episodes):
                print("Running episode ", ep + 1, " of ", num_episodes, "; policy = ", policy_type,
                      " ; total time elapsed ", time.time() - t)
                _ = self.reset()
                done = False
                while not done:
                    ac = self.obtain_baseline_policy(policy_type)
                    if isinstance(self.action_space, spaces.Box):
                        ac = np.clip(ac, self.action_space.low, self.action_space.high)
                    _, _, done, info = self.step(ac)
                total_rewards[ep] = info["mean_total_rwd"]
        elif policy_type == 'grid':
            acs = self.obtain_baseline_policy(policy_type)  # Action is constant!!
            for ep in range(num_episodes):
                print("Running episode ", ep + 1, " of ", num_episodes, "; policy = ", policy_type,
                      " ; total time elapsed ", time.time() - t)
                _ = self.reset()
                done = False
                while not done:
                    if self.attack_mode == 'phy':
                        action = np.clip(acs, self.action_space.low, self.action_space.high)  # Action is continuous
                        action = np.array([action for _ in range(self.nr_agents)])  # Repeat action!
                    elif self.attack_mode == 'mac':
                        action = np.random.binomial(1, acs, self.nr_agents)  # Action is discrete!
                    _, _, done, info = self.step(action)
                total_rewards[ep] = info["mean_total_rwd"]
        return total_rewards

    def obtain_baseline_policy(self, policy_type):
        # Obtains the baseline action to compare the learning results
        if policy_type == 'random':
            return [self.action_space.sample() for _ in range(self.nr_agents)]
        elif policy_type == 'grid':
            return self.get_grid_action()
        else:
            raise RuntimeError('Baseline policy type not recognized')

    def get_grid_action(self, max_actions=10, n_rep=5):
        av = np.linspace(0, 1, max_actions).tolist()  # Generate the grid
        reward = np.zeros(max_actions)
        for ac in av:
            if isinstance(self.action_space, spaces.Box):
                action = np.clip(ac, self.action_space.low, self.action_space.high)  # Action is continuous
                action = np.array([action for _ in range(self.nr_agents)])  # Repeat action!
            elif isinstance(self.action_space, spaces.Discrete):
                action = np.random.binomial(1, ac, self.nr_agents)  # Action is discrete!
            else:
                raise RuntimeError('Action space not valid')

            # Simulate to obtain the reward expected
            for _ in range(n_rep):
                _ = self.reset()
                done = False
                while not done:
                    _, _, done, info = self.step(action)
                reward[av.index(ac)] += info["mean_total_rwd"]/n_rep

        return av[np.argmax(reward)]  # Maximum action


class random_pol():  # This classes are only created for debugging

    def __init__(self, env):
        self.env = env

    def act(self, stochastic, state):
        av = [self.env.action_space.sample() for _ in range(self.env.nr_agents)]
        return np.array(av), None

class mac_neutral_policy():
    def __init__(self):
        self.action = []

    def act(self, ob, stochastic=True):
        return self.action, None

if __name__ == '__main__':
    n_ag = 0
    env = generate_env('attack').generate()
    def_mech = defense_mechanism('reward', './expert_data/0/reward_giver', './expert_data/0/trajs_neutral.npz', env,
                                 n_agents=10)

    tl = [5, 10, 20, 30, 40, 50]
    rep = 50
    error = np.zeros((len(tl), rep))
    for trace_length in tl:
        env = AttackEnv(nr_agents=n_ag, nr_agents_total=10, obs_mode='sum_obs', attack_mode="mac",
                        obs_radius=5000, world_size=1000, phy_agent_memory=5, lambda_phy=0.5,
                        K=5, L=1000, lambda_mac=0.5, timesteps_limit=250, sum_rwd=False, df=0.995,
                        def_mech=def_mech, def_trace_length=20)

        episodes = rep
        rt = 0
        #pol = mac_neutral_policy()
        for e in range(episodes):
            print("episode ", e + 1)
            o = env.reset()
            done = False
            while not done:
                a = []
                obs, rew, done, info = env.step(a)
                rt += np.sum(rew)
            error[tl.index(trace_length), e] = np.sum(env.banned) / 10
    print('Error = ', np.mean(error, axis=1))
    import matplotlib.pyplot as plt
    bins = np.linspace(0, 15, 100)
    for s in range(env.nr_agents_total):
        plt.hist(env.aux[s][0], bins, alpha=0.5, label=str(s))
    plt.legend(loc='best')
    plt.show()
    for s in range(env.nr_agents_total):
        plt.hist(env.aux[s][-1], bins, alpha=0.5, label=str(s))
    plt.legend(loc='best')
    plt.show()
    print('end of simulation')