import numpy as np
import pickle


def traj_1_generator(pi, env, horizon, stochastic, mode):
    # Obtain 1 trajectory (max length can be adjusted in horizon)
    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()
    if isinstance(ob, list):
        ob = np.squeeze(np.array(ob))
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
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
        # FIXME: Here, ac and vpred are np arrays, should they be be passed as numbers?
        if isinstance(ac, np.ndarray) and env.nr_agents == 1:
            ac = ac[0]
        obs.append(ob)
        news.append(new)
        acs.append(ac)

        ob, rew, new, info = env.step(ac)
        if isinstance(ob, list) and env.nr_agents == 1:
            ob = np.squeeze(np.array(ob))
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1

    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len, "info": info}
    return traj


class PolicyEvaluator():
    def __init__(self, env, policy, filename, n_evals=100, mode='step'):
        from tqdm import tqdm
        print('Evaluating policy...')
        obs_list = []
        acs_list = []
        len_list = []
        ret_list = []
        info = []
        # Obtain expert trajectories
        for _ in tqdm(range(n_evals)):
            traj = traj_1_generator(policy, env, np.inf, True, mode)
            obs, acs, ep_len, ep_ret, infor = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret'], traj["info"]
            obs_list.append(obs)
            acs_list.append(acs)
            len_list.append(ep_len)
            ret_list.append(ep_ret)
            info.append(infor)
        # Save expert trajectories
        #np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
        #         lens=np.array(len_list), ep_rets=np.array(ret_list))
        if info is not None:
            with open(filename + '.pickle', 'wb') as handle:
                pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)