class generate_env():
    def __init__(self, name, na=1, nt=11):
        self.name = name
        self.na = na # Number of agents attackers
        self.nt = nt # Total number of sensors

    def generate(self, reward_path=None, neutral_data=None, seed=None):
        if self.na > 1:
            obs_radius = 5000
        else:
            obs_radius = 0

        if self.name == 'CartPole-v0':
            import gym
            return gym.make('CartPole-v0')
        elif self.name == 'attack':
            from attack import AttackEnv
            return AttackEnv(nr_agents=self.na,  # Agents in the swarm
                             nr_agents_total=self.nt,  # Total number of agents (i.e., AS + NS)
                             obs_mode='sum_obs_multi', # Observation mode
                             attack_mode='mac', # PHY or MAC attack
                             obs_radius=obs_radius, # Observation radius of the AS
                             world_size=1000, # Size of the world
                             K=5,  # MAC attack parameter
                             L=1000,  # MAC defense parameter
                             lambda_mac=0.5, # MAC attack threshold (-1 for random changing)
                             timesteps_limit=400,  # Max number of timesteps for the attack
                             sum_rwd=False, # If true, all ASs have as reward the sum of the rewards of all ASs
                             df=0.995)  # To obtain discounted reward
        elif self.name == 'attack_neutral':
            from attack import AttackEnv
            return AttackEnv(nr_agents=0,  # No agent: this is used for obtaining neutral values
                             nr_agents_total=self.nt,  # Total number of agents (i.e., AS + NS)
                             obs_mode='sum_obs_multi', # Observation mode
                             obs_returned='def', # Returns the observation of a all agents
                             attack_mode='mac', # PHY or MAC attack
                             obs_radius=0, # Observation radius of the AS
                             world_size=1000, # Size of the world
                             K=5,  # MAC attack parameter
                             L=1000,  # MAC defense parameter
                             lambda_mac=0.5, # MAC attack threshold (-1 for random changing)
                             timesteps_limit=400,  # Max number of timesteps for the attack
                             sum_rwd=False, # If true, all ASs have as reward the sum of the rewards of all ASs
                             df=0.995 # To obtain discounted reward
                             #nr_neutral_agents=1)
                             )
        elif self.name == 'attack_for_gail':
            from attack import AttackEnv
            return AttackEnv(nr_agents=self.nt,  # Used to train gail
                             nr_agents_total=self.nt,  # Total number of agents (i.e., AS + NS)
                             obs_mode='sum_obs_multi', # Observation mode
                             obs_returned='def', # Returns the observation of a all agents
                             attack_mode='mac', # PHY or MAC attack
                             obs_radius=0, # Observation radius of the AS
                             world_size=1000, # Size of the world
                             K=5,  # MAC attack parameter
                             L=1000,  # MAC defense parameter
                             lambda_mac=0.5, # MAC attack threshold (-1 for random changing)
                             timesteps_limit=400,  # Max number of timesteps for the attack
                             sum_rwd=False, # If true, all ASs have as reward the sum of the rewards of all ASs
                             df=0.995 # To obtain discounted reward
                             #nr_neutral_agents=1)
                             )
        elif self.name == 'attack_and_defense':
            from attack import AttackEnv
            from defense import defense_mechanism

            env =  AttackEnv(nr_agents=self.na,  # Agents in the swarm
                             nr_agents_total=self.nt,  # Total number of agents (i.e., AS + NS)
                             obs_mode='sum_obs_multi', # Observation mode
                             attack_mode='mac', # PHY or MAC attack
                             obs_radius=obs_radius, # Observation radius of the AS
                             world_size=1000, # Size of the world
                             K=5,  # MAC attack parameter
                             L=1000,  # MAC defense parameter
                             lambda_mac=0.5, # MAC attack threshold (-1 for random changing)
                             timesteps_limit=400,  # Max number of timesteps for the attack
                             sum_rwd=False, # If true, all ASs have as reward the sum of the rewards of all ASs
                             df=0.995)  # To obtain discounted reward
            def_mech = defense_mechanism('reward', reward_path, neutral_data, env, n_agents=self.nt)
            return AttackEnv(nr_agents=self.na,  # Agents in the swarm
                             nr_agents_total=self.nt,  # Total number of agents (i.e., AS + NS)
                             obs_mode='sum_obs_multi', # Observation mode
                             attack_mode='mac', # PHY or MAC attack
                             obs_radius=obs_radius, # Observation radius of the AS
                             world_size=1000, # Size of the world
                             K=5,  # MAC attack parameter
                             L=1000,  # MAC defense parameter
                             lambda_mac=0.5, # MAC attack threshold (-1 for random changing)
                             timesteps_limit=400,  # Max number of timesteps for the attack
                             sum_rwd=False, # If true, all ASs have as reward the sum of the rewards of all ASs
                             df=0.995, # To obtain discounted reward
                             def_mech=def_mech, # Defense mechanism
                             def_trace_length=5 # For defense mechanism
                             )
        elif self.name == 'attack_and_defense_online':
            from attack import AttackEnv
            from defense import defense_mechanism_online

            env =  AttackEnv(nr_agents=self.na,  # Agents in the swarm
                             nr_agents_total=self.nt,  # Total number of agents (i.e., AS + NS)
                             obs_mode='sum_obs_multi', # Observation mode
                             attack_mode='mac', # PHY or MAC attack
                             obs_radius=obs_radius, # Observation radius of the AS
                             world_size=1000, # Size of the world
                             K=5,  # MAC attack parameter
                             L=1000,  # MAC defense parameter
                             lambda_mac=0.5, # MAC attack threshold (-1 for random changing)
                             timesteps_limit=400,  # Max number of timesteps for the attack
                             sum_rwd=False, # If true, all ASs have as reward the sum of the rewards of all ASs
                             df=0.995)  # To obtain discounted reward
            def_mech = defense_mechanism_online('reward', reward_path, neutral_data, env, n_agents=self.nt, nat=self.na,
                                                seed=seed)
            return AttackEnv(nr_agents=self.na,  # Agents in the swarm
                             nr_agents_total=self.nt,  # Total number of agents (i.e., AS + NS)
                             obs_mode='sum_obs_multi', # Observation mode
                             attack_mode='mac', # PHY or MAC attack
                             obs_radius=obs_radius, # Observation radius of the AS
                             world_size=1000, # Size of the world
                             K=5,  # MAC attack parameter
                             L=1000,  # MAC defense parameter
                             lambda_mac=0.5, # MAC attack threshold (-1 for random changing)
                             timesteps_limit=400,  # Max number of timesteps for the attack
                             sum_rwd=False, # If true, all ASs have as reward the sum of the rewards of all ASs
                             df=0.995, # To obtain discounted reward
                             def_mech=def_mech, # Defense mechanism
                             def_trace_length=5 # For defense mechanism
                             )
        else:
            raise RuntimeError('Environment not recognized')