'''
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
'''

from baselines import logger
import numpy as np


class Dset(object):
    def __init__(self, inputs, labels, randomize, n_agents=0):
        if n_agents == 0:
            self.inputs = inputs
            self.labels = labels
        else:
            self.inputs = np.concatenate(inputs)
            self.labels = np.concatenate(labels)
        assert len(self.inputs) == len(self.labels)
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels


class Mujoco_Dset(object):
    def __init__(self, expert_path, train_fraction=0.7, traj_limitation=-1, randomize=True, n_agents=0):
        traj_data = np.load(expert_path, allow_pickle=True)
        if traj_limitation < 0:
            traj_limitation = len(traj_data['obs'])
        if n_agents == 0:
            obs = traj_data['obs'][:traj_limitation]
            acs = traj_data['acs'][:traj_limitation]

            # FIXME: This is needed for action spaces with 1 dimension:
            if len(acs[0].shape) == 1:
                for i in range(len(acs)):
                    acs[i] = acs[i].reshape([len(acs[i]), 1])
            # obs, acs: shape (N, L, ) + S where N = # episodes, L = episode length
            # and S is the environment observation/action space.
            # Flatten to (N * L, prod(S))
            if len(obs.shape) > 2:
                self.obs = np.reshape(obs, [-1, np.prod(obs.shape[2:])])
                self.acs = np.reshape(acs, [-1, np.prod(acs.shape[2:])])
            else:
                self.obs = np.vstack(obs)
                self.acs = np.vstack(acs)
            if 'done' in traj_data.files:
                self.done = traj_data['done'][:traj_limitation]
            else:
                self.done = None
            self.rets = traj_data['ep_rets'][:traj_limitation]
            self.avg_ret = sum(self.rets)/len(self.rets)
            self.std_ret = np.std(np.array(self.rets))
            # TO DO: this line has been commented out for problems with 1 D actions
            #if len(self.acs) > 2:
            #    self.acs = np.squeeze(self.acs)
            assert len(self.obs) == len(self.acs)
            self.num_traj = min(traj_limitation, len(traj_data['obs']))
            self.num_transition = len(self.obs)
            self.randomize = randomize
            self.dset = Dset(self.obs, self.acs, self.randomize)
            # for behavior cloning
            self.train_set = Dset(self.obs[:int(self.num_transition*train_fraction), :],
                                  self.acs[:int(self.num_transition*train_fraction), :],
                                  self.randomize)
            self.val_set = Dset(self.obs[int(self.num_transition*train_fraction):, :],
                                self.acs[int(self.num_transition*train_fraction):, :],
                                self.randomize)
            self.log_info()
        else:
            self.obs = [[] for _ in range(n_agents)]
            self.acs = [[] for _ in range(n_agents)]
            self.rets = [[] for _ in range(n_agents)]
            self.avg_ret = [[] for _ in range(n_agents)]
            self.std_ret = [[] for _ in range(n_agents)]
            self.done = [[] for _ in range(n_agents)]
            for agent in range(n_agents):
                obs = np.concatenate(traj_data['obs'][agent, :traj_limitation])
                acs = np.concatenate(traj_data['acs'][agent, :traj_limitation])

                # FIXME: This is needed for action spaces with 1 dimension:
                if len(acs[0].shape) == 1:
                    for i in range(len(acs)):
                        acs[i] = acs[i].reshape([len(acs[i]), 1])
                # obs, acs: shape (N, L, ) + S where N = # episodes, L = episode length
                # and S is the environment observation/action space.
                # Flatten to (N * L, prod(S))
                if len(obs.shape) > 2:
                    self.obs[agent] = np.reshape(obs, [-1, np.prod(obs.shape[2:])])
                    self.acs[agent] = np.reshape(acs, [-1, np.prod(acs.shape[2:])])
                else:
                    self.obs[agent] = np.vstack(obs)
                    self.acs[agent] = np.vstack(acs)
                if 'done' in traj_data.files:
                    self.done[agent] = traj_data['done'][:traj_limitation]
                else:
                    self.done[agent] = None
                self.rets[agent] = np.concatenate(traj_data['ep_rets'][:traj_limitation])
                self.avg_ret[agent] = sum(self.rets[agent]) / len(self.rets[agent])
                self.std_ret[agent] = np.std(np.array(self.rets[agent]))
                # TO DO: this line has been commented out for problems with 1 D actions
                # if len(self.acs) > 2:
                #    self.acs = np.squeeze(self.acs)
                assert len(self.obs[agent]) == len(self.acs[agent])
            self.num_traj = np.prod(traj_data['obs'].shape)
            self.num_transition = sum([len(self.obs[agent]) for agent in range(n_agents)])
            self.randomize = randomize

            self.dset = Dset(self.obs, self.acs, self.randomize, n_agents=n_agents)
            '''
            # for behavior cloning
            self.train_set = Dset(self.obs[:int(self.num_transition * train_fraction), :],
                                  self.acs[:int(self.num_transition * train_fraction), :],
                                  self.randomize)
            self.val_set = Dset(self.obs[int(self.num_transition * train_fraction):, :],
                                self.acs[int(self.num_transition * train_fraction):, :],
                                self.randomize)
            
            self.log_info()
            '''

    def log_info(self):
        logger.log("Total trajectorues: %d" % self.num_traj)
        logger.log("Total transitions: %d" % self.num_transition)
        logger.log("Average returns: %f" % self.avg_ret)
        logger.log("Std for returns: %f" % self.std_ret)

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()


def test(expert_path, traj_limitation, plot):
    dset = Mujoco_Dset(expert_path, traj_limitation=traj_limitation)
    if plot:
        dset.plot()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="../data/deterministic.trpo.Hopper.0.00.npz")
    parser.add_argument("--traj_limitation", type=int, default=None)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()
    test(args.expert_path, args.traj_limitation, args.plot)
