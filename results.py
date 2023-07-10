import numpy as np
import pickle
from generate_env import generate_env


def checkout(policy_path, env_id, reward_giver_path=None, expert_path=None):

    if env_id == 'attack_and_defense':
        env = generate_env(env_id).generate(reward_path=reward_giver_path, neutral_data=expert_path)
    else:
        env = generate_env(env_id).generate()
    from baselines.common.policies import build_policy
    from baselines.common.input import observation_placeholder
    import baselines.common.tf_util as U
    from baselines.trpo_mpi.trpo_mpi import  traj_segment_generator
    import tensorflow as tf

    policy = build_policy(env, 'mlp', value_network='copy', num_hidden=256, activation=tf.nn.relu)
    ob_space = env.observation_space
    ob = observation_placeholder(ob_space)
    with tf.variable_scope("pi"):
        pi = policy(observ_placeholder=ob)
    U.initialize()
    pi.load(policy_path)
    seg_gen = traj_segment_generator(pi, env, 1000, stochastic=False)
    seg = seg_gen.__next__()


def generate_distributions():
    from attack import AttackEnv
    from defense import defense_mechanism
    import os
    nat = 10

    env = AttackEnv(nr_agents=nat,  # Agents in the swarm
                    nr_agents_total=nat + 10,  # Total number of agents (i.e., AS + NS)
                    obs_mode='sum_obs_multi',  # Observation mode
                    attack_mode='mac',  # PHY or MAC attack
                    obs_radius=5000,  # Observation radius of the AS
                    world_size=1000,  # Size of the world
                    K=5,  # MAC attack parameter
                    L=1000,  # MAC defense parameter
                    lambda_mac=0.5,  # MAC attack threshold (-1 for random changing)
                    timesteps_limit=400,  # Max number of timesteps for the attack
                    sum_rwd=False,  # If true, all ASs have as reward the sum of the rewards of all ASs
                    df=0.995)  # To obtain discounted reward
    def_mech = defense_mechanism('reward', os.path.join(os.getcwd(), 'expert_data', str(nat), str(0), 'reward_giver'),
                                 os.path.join(os.getcwd(), 'expert_data', str(nat), str(0), 'trajs_neutral.npz'), env,
                                 n_agents=nat+10)

    env =  AttackEnv(nr_agents=nat,  # Agents in the swarm
                     nr_agents_total=nat+10,  # Total number of agents (i.e., AS + NS)
                     obs_mode='sum_obs_multi',  # Observation mode
                     attack_mode='mac',  # PHY or MAC attack
                     obs_radius=5000,  # Observation radius of the AS
                     world_size=1000,  # Size of the world
                     K=5,  # MAC attack parameter
                     L=1000,  # MAC defense parameter
                     lambda_mac=0.5,  # MAC attack threshold (-1 for random changing)
                     timesteps_limit=400,  # Max number of timesteps for the attack
                     sum_rwd=False,  # If true, all ASs have as reward the sum of the rewards of all ASs
                     df=0.995,  # To obtain discounted reward
                     def_mech=def_mech,  # Defense mechanism
                     def_trace_length=5  # For defense mechanism
                     )

if __name__ == "__main__":
    file_de = './results_de/5/0/'
    file_nd = './results_nd/5/0/'

    with open(file_de + 'policy_results.pickle', "rb") as input_file:
        data_de = pickle.load(input_file)

    with open(file_nd + 'policy_results.pickle', "rb") as input_file:
        data_nd = pickle.load(input_file)

    def data_extractor(data1, data2, title1, title2):
        for key in data1[0].keys():
            if isinstance(data1[0][key], list):
                pass
            else:
                mean1 = 0
                mean2 = 0
                ndata = len(data1)
                for i in range(ndata):
                    mean1 += data1[i][key] / ndata
                    mean2 += data2[i][key] / ndata
                print(title1, ' - ', str(key), ' = ', str(mean1), ' || ', title2, ' - ', str(key), ' = ', str(mean2))
    data_extractor(data_nd, data_de, 'nd', 'de')
    '''
    checkout('./results_de/1/policy', 'attack_and_defense',
             reward_giver_path='./expert_data/1/reward_giver',
             expert_path='./expert_data/1/trajs_neutral.npz')
    '''
    print('done')