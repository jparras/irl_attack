import os
from generate_env import generate_env
from attack import AttackEnv
from baselines.gail.dataset.mujoco_dset import Mujoco_Dset
from baselines.gail.adversary import TransitionClassifier
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
#from matplotlib2tikz import save as tikz_save
from tikzplotlib import save as tikz_save
import argparse
from sklearn.neighbors import KernelDensity


def argsparser():
    parser = argparse.ArgumentParser("Obtain histogram plots")
    parser.add_argument('--env_id', help='environment ID', default='attack_and_defense_online')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='./expert_data_online/1/0/trajs_neutral.npz')
    parser.add_argument('--reward_path', type=str, default='./expert_data_online/1/0/reward_giver')
    parser.add_argument('--results_path', type=str, default='./results_de_online/1/0/policy')
    parser.add_argument('--mode', type=str, default='online')
    parser.add_argument('--nat', help='AS', type=int, default=1)
    parser.add_argument('--nt', help='GS+AS', type=int, default=11)

    return parser.parse_args()

def traj_1_generator(pi, env, horizon, stochastic, mode):
    # Obtain 1 trajectory (max length can be adjusted in horizon)
    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()
    ob_def = env.get_def_obs()
    if isinstance(ob, list):
        ob = np.squeeze(np.array(ob))
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    obs_def = []
    rews = []
    news = []
    acs = []

    while True:
        if mode == 'step':
            ac, vpred, _, _ = pi.step(ob, stochastic=stochastic)
        elif mode == 'act':
            ac, vpred = pi.act(stochastic, ob)
        else:
            raise RuntimeError('Policy mode not recognized')
        # FIXME: Here, ac and vpred are np arrays, should they be numbers?
        if isinstance(ac, np.ndarray) and env.nr_agents == 1:
            ac = ac[0]
        obs.append(ob)
        obs_def.append(ob_def)
        news.append(new)

        ob, rew, new, info = env.step(ac)
        ob_def = env.get_def_obs()
        acs.append(info['ac_list'])  # ac_list includes the actions of the gs
        if isinstance(ob, list) and env.nr_agents == 1:
            ob = np.squeeze(np.array(ob))
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1

    obs = np.array(obs)
    obs_def = np.array(obs_def)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    traj = {"ob": obs_def, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len, "info": info}
    return traj


def obtain_reward_gs(args, def_alpha=0.05, hidden_size=256, entcoeff=1e-3):

    if args['nat'] > 1:
        obs_radius = 5000
    else:
        obs_radius = 0

    env = AttackEnv(nr_agents=args['nat'],  # Agents in the swarm
                    nr_agents_total=args['nt'],  # Total number of agents (i.e., AS + NS)
                    obs_mode='sum_obs_multi',  # Observation mode
                    attack_mode='mac',  # PHY or MAC attack
                    obs_radius=obs_radius,  # Observation radius of the AS
                    world_size=1000,  # Size of the world
                    K=5,  # MAC attack parameter
                    L=1000,  # MAC defense parameter
                    lambda_mac=0.5,  # MAC attack threshold (-1 for random changing)
                    timesteps_limit=400,  # Max number of timesteps for the attack
                    sum_rwd=False,  # If true, all ASs have as reward the sum of the rewards of all ASs
                    df=0.995)  # To obtain discounted reward
    if args['env_id'] == 'attack_and_defense':
        from defense import defense_mechanism
        def_mech = defense_mechanism('reward', args['reward_path'], args['expert_path'], env, n_agents=args['nt'])
    elif args['env_id'] == 'attack_and_defense_online':
        from defense import defense_mechanism_online
        def_mech = defense_mechanism_online('reward', args['reward_path'], args['expert_path'], env, n_agents=args['nt'])
        def_mech.load_def()
    else:
        raise NotImplementedError

    dataset_neutral = Mujoco_Dset(expert_path=args['expert_path'], traj_limitation=-1, n_agents=5)  # Use only 5 agents
    if args['nt'] == 0:
        # Obtain the rewards for each state - action pair
        num_transitions = dataset_neutral.obs.shape[0]
        rws = np.zeros(num_transitions)
        for i in range(num_transitions):
            rws[i] = def_mech.def_model.get_reward(dataset_neutral.obs[i], dataset_neutral.acs[i])
    else:
        assert 5 == len(dataset_neutral.obs)
        num_transitions = dataset_neutral.obs[0].shape[0]
        rws = np.zeros((5, num_transitions))
        for ag in range(5):
            for i in range(num_transitions):
                rws[ag, i] = def_mech.def_model.get_reward(dataset_neutral.obs[ag][i], dataset_neutral.acs[ag][i])
    rws = rws.flatten()
    #sorted_data = np.sort(rws)
    #cdf = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
    #def_threshold = sorted_data[np.where(cdf >= def_alpha)[0][0] - 1]

    return rws, def_mech.def_threshold, def_mech


def obtain_reward_as(args, def_model):

    # Create env
    if args['env_id'] == 'attack_and_defense':
        if args['nat'] > 1:
            obs_radius = 5000
        else:
            obs_radius = 0
        env =  AttackEnv(nr_agents=args['nat'],  # Agents in the swarm
                         nr_agents_total=args['nt'],  # Total number of agents (i.e., AS + NS)
                         obs_mode='sum_obs_multi',  # Observation mode
                         attack_mode='mac',  # PHY or MAC attack
                         obs_radius=obs_radius,  # Observation radius of the AS
                         world_size=1000,  # Size of the world
                         K=5,  # MAC attack parameter
                         L=1000,  # MAC defense parameter
                         lambda_mac=0.5,  # MAC attack threshold (-1 for random changing)
                         timesteps_limit=400,  # Max number of timesteps for the attack
                         sum_rwd=False,  # If true, all ASs have as reward the sum of the rewards of all ASs
                         df=0.995,  # To obtain discounted reward
                         def_mech=def_model,  # Defense mechanism
                         def_trace_length=5  # For defense mechanism
                         )
    elif args['env_id'] == 'attack_and_defense_online':
        if args['nat'] > 1:
            obs_radius = 5000
        else:
            obs_radius = 0
        env = AttackEnv(nr_agents=args['nat'],  # Agents in the swarm
                        nr_agents_total=args['nt'],  # Total number of agents (i.e., AS + NS)
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
                        def_mech=def_model, # Defense mechanism
                        def_trace_length=5 # For defense mechanism
                        )
    else:
        raise NotImplementedError

    if args['nat'] > 1:
        from obtain_attack_results import MlpPolicy_Multi_Mean_Embedding

        def policy(name, ob_space, ac_space, index=None):
            return MlpPolicy_Multi_Mean_Embedding(name=name, ob_space=ob_space, ac_space=ac_space,
                                                  hid_size=[256, 256], feat_size=[[], []], index=index)
    else:
        from obtain_attack_results import MlpPolicy_No_Mean_Emmbedding

        def policy(name, ob_space, ac_space, index=None):
            return MlpPolicy_No_Mean_Emmbedding(name=name, ob_space=ob_space, ac_space=ac_space,
                                                hid_size=[256, 256], feat_size=[256], index=index)
    from obtain_attack_results import load_policy
    expert_policy = load_policy(env, policy, args['results_path'])

    obs_list = []
    acs_list = []
    # Obtain expert trajectories
    for _ in range(5):
        traj = traj_1_generator(expert_policy, env, np.inf, True, 'act')
        obs, acs = traj['ob'], traj['ac']
        obs_list.append(obs)
        acs_list.append(acs)

    # Obtain the rewards for each state - action pair
    rws_as = []
    rws_gs = []
    for i in range(len(obs_list)): # Number of episodes
        for j in range(obs_list[i].shape[0]): # Time
            for k in range(args['nt']):
                if k < args['nat']:
                    rws_as.append(def_model.def_model.get_reward(obs_list[i][j,k], acs_list[i][j,k]))
                else:
                    rws_gs.append(def_model.def_model.get_reward(obs_list[i][j,k], acs_list[i][j,k]))

    return np.array(rws_as).flatten(), np.array(rws_gs).flatten()


if __name__ == "__main__":
    arg = argsparser()
    args = {
        'env_id': arg.env_id,
        'seed': arg.seed,
        'expert_path': arg.expert_path,
        'reward_path': arg.reward_path,
        'results_path': arg.results_path,
        'nat': arg.nat,
        'nt': arg.nt
    }
    reward_gs, def_threshold, def_model = obtain_reward_gs(args)
    reward_as, reward_gs_attack = obtain_reward_as(args, def_model)

    nbins = 100
    bmin = min([np.amin(reward_as), np.amin(reward_gs), np.amin(reward_gs_attack)])
    bmax = max([np.amax(reward_as), np.amax(reward_gs), np.amax(reward_gs_attack)])

    bins = np.linspace(bmin, bmax, nbins)

    # Create folder to store results
    if not os.path.exists('./results'):
        os.makedirs('./results')

    # KDE
    model = KernelDensity(bandwidth=0.02)
    model.fit(reward_gs[:, np.newaxis])
    plt.plot(bins, np.exp(model.score_samples(bins[:, np.newaxis])), label='GS NA')
    model.fit(reward_gs_attack[:, np.newaxis])
    plt.plot(bins, np.exp(model.score_samples(bins[:, np.newaxis])), label='GS UA')
    model.fit(reward_as[:, np.newaxis])
    plt.plot(bins, np.exp(model.score_samples(bins[:, np.newaxis])), label='AS')
    plt.axvline(x=def_threshold)
    plt.legend(loc='best')
    title = str(args['nat']) + '_' + str(args['seed']) + '_' + str(arg.mode)
    plt.title(title)
    tikz_save(title + '_d.tikz', encoding='utf-8')
    plt.savefig(title + '_d.png', bbox_inches='tight')

    plt.hist(reward_gs, bins, alpha=0.5, label='GS NA', density=True)
    plt.hist(reward_gs_attack, bins, alpha=0.5, label='GS UA', density=True)
    plt.hist(reward_as, bins, alpha=0.5, label='AS', density=True)
    plt.axvline(x=def_threshold)
    plt.legend(loc='best')
    title = str(args['nat']) + '_' + str(args['seed']) + '_' + str(arg.mode)
    plt.title(title)
    tikz_save('./results/' + title + '.tikz', encoding='utf-8')
    plt.savefig('./results/' + title + '.png', bbox_inches='tight')


