import numpy as np


def traj_1_generator(env, horizon, stochastic):
    # Obtain 1 trajectory (max length can be adjusted in horizon)
    t = 0
    new = True  # marks if we're on first timestep of an episode

    _ = env.reset()
    ob = env.get_def_obs()
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []
    dones = []

    while True:
        obs.append(ob)
        news.append(new)

        _, rew, new, info = env.step([])  # Neutral policy: empty list for actions!
        ac = info['ac_list']
        ob = info['def_obs']
        acs.append(ac)
        rews.append(rew)
        dones.append(new)

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
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len, "done": dones}

    return traj

class Generator(object):
    def __init__(self, env, n_trajs, st_index=None, save_path=None):
        # Obtain expert trajectories
        n_traj = 0
        while n_traj < n_trajs:
            traj = traj_1_generator(env, np.inf, False)
            agents = range(traj['ob'].shape[1])

            if st_index is not None:
                agents = [st_index]

            if n_traj == 0:  #Initialize values to save
                obs_list = [[] for _ in agents]
                acs_list = [[] for _ in agents]
                done_list = [[] for _ in agents]
                len_list = [[] for _ in agents]
                ret_list = [[] for _ in agents]

            for agent in agents:
                ret = 0  # Default value
                obs = np.squeeze(traj['ob'][:, agent, :])
                acs = np.expand_dims(traj['ac'][:, agent], 1)
                index = agents.index(agent)
                obs_list[index].append(obs)
                acs_list[index].append(acs)
                ret_list[index].append(ret)
                done_list[index].append(traj['done'])
                len_list[index].append(traj['ep_len'])
                n_traj += 1  # We add an agent trajectory

        if save_path is not None:
            # Save expert trajectories
            np.savez(save_path, obs=np.array(obs_list), acs=np.array(acs_list), done=np.array(done_list),
                     lens=np.array(len_list), ep_rets=np.array(ret_list))


class mac_neutral_policy():
    def __init__(self):
        self.action = []

    def step(self, ob, stochastic=True):
        return self.action, None, None, None

if __name__ == "__main__":

    # Create env
    from generate_env import generate_env
    ge = generate_env('attack_neutral')
    env = ge.generate()

    Generator(env, n_trajs=20, save_path='./data.npz')

