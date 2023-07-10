import platform
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Uncomment this to run on CPU (needed if n_threads > 1)
import numpy as np
from joblib import Parallel, delayed
from generate_env import generate_env
import pickle
import itertools
import matplotlib.pyplot as plt
#from matplotlib2tikz import save as tikz_save
from tikzplotlib import save as tikz_save
from scipy import stats


if __name__ == "__main__":
    # Set up parameters
    nr_seeds = 10
    n_threads = 20
    plot_final = True
    nr_seeds_plot = 5  # Number of seeds to plot
    obtain_histograms = True  # To obtain histograms of the results
    train = False  # To train the networks (takes time)
    nr_at = [1, 5, 10]  # Number of AS
    nr_ns = 10  # Number of normal sensors
    # Neutral data related configuration
    n_trajs = 100  # Number of neutral trajectories to store (offline case)

    # TRPO attack parameters
    trpo_iters = 200

    def process(seed, nat):
        orders = []
        # Create directory to store info
        dir = os.path.join(os.getcwd(), 'expert_data', str(nat), str(seed))
        if not os.path.exists(dir):
            os.makedirs(dir)

        # First, generate a set of sample trajectories using an environment without attackers
        from obtain_neutral_values import Generator
        env = generate_env('attack_neutral', na=nat, nt=nat + nr_ns).generate()  # Neutral environment used for obtaining neutral trajs
        save_path = os.path.join(os.getcwd(), 'expert_data', str(nat), str(seed), 'trajs_neutral.npz')
        if not os.path.isfile(save_path):
            Generator(env, n_trajs, save_path=save_path)


        # Train the reward function using GAIL
        expert_path = os.path.join(os.getcwd(), 'expert_data', str(nat), str(seed), 'trajs_neutral.npz')
        rew_dir = os.path.join(os.getcwd(), 'expert_data', str(nat), str(seed), 'reward_giver')  # To save reward_giver
        ga_rw = 'generate_gail_classifier.py --env_id=' + 'attack_for_gail' + \
                ' --seed=' + str(seed) + \
                ' --expert_path=' + str(expert_path) + \
                ' --reward_giver_path=' + str(rew_dir) + \
                ' --g_step=' + str(3) + \
                ' --d_step=' + str(1) + \
                ' --num_iters=' + str(10) + \
                ' --nat=' + str(nat) + \
                ' --ns=' + str(nr_ns)
        if not os.path.isfile(rew_dir + '.meta'):
            orders.append(ga_rw)

        # Obtain the results of an attack without defense mechanism
        res_path = os.path.join(os.getcwd(), 'results_nd', str(nat), str(seed))
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        res_path += '/'
        at_nd = 'obtain_attack_results.py --env_id=' + 'attack' + \
                ' --seed=' + str(seed) + \
                ' --expert_path=' + str(expert_path) + \
                ' --reward_giver_path=' + str(rew_dir) + \
                ' --training_iters=' + str(trpo_iters) + \
                ' --results_path=' + res_path + \
                ' --nat=' + str(nat) + \
                ' --ns=' + str(nr_ns)
        if not os.path.isfile(res_path + 'training_results.pickle'):
            orders.append(at_nd)

        # Obtain the results of an attack with offline defense mechanism
        res_path = os.path.join(os.getcwd(), 'results_de', str(nat), str(seed))
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        res_path += '/'
        at_de = 'obtain_attack_results.py --env_id=' + 'attack_and_defense' + \
                ' --seed=' + str(seed) + \
                ' --expert_path=' + str(expert_path) + \
                ' --reward_giver_path=' + str(rew_dir) + \
                ' --training_iters=' + str(trpo_iters) + \
                ' --results_path=' + res_path + \
                ' --nat=' + str(nat) + \
                ' --ns=' + str(nr_ns)
        if not os.path.isfile(res_path + 'training_results.pickle'):
            orders.append(at_de)
        # Obtain the results of an attack with online defense mechanism
        expert_path_online = os.path.join(os.getcwd(), 'expert_data_online', str(nat), str(seed), 'trajs_neutral.npz')
        rew_dir_online = os.path.join(os.getcwd(), 'expert_data_online', str(nat), str(seed), 'reward_giver')  # To save reward_giver
        res_path = os.path.join(os.getcwd(), 'results_de_online', str(nat), str(seed))
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        if not os.path.exists(os.path.join(os.getcwd(), 'expert_data_online', str(nat), str(seed))):
            os.makedirs(os.path.join(os.getcwd(), 'expert_data_online', str(nat), str(seed)))
        res_path += '/'
        at_do = 'obtain_attack_results.py --env_id=' + 'attack_and_defense_online' + \
                ' --seed=' + str(seed) + \
                ' --expert_path=' + str(expert_path_online) + \
                ' --reward_giver_path=' + str(rew_dir_online) + \
                ' --training_iters=' + str(trpo_iters) + \
                ' --results_path=' + res_path + \
                ' --nat=' + str(nat) + \
                ' --ns=' + str(nr_ns)
        if not os.path.isfile(res_path + 'training_results.pickle'):
            orders.append(at_do)

        for order in orders:
            if platform.system() == 'Windows':
                print("Running on Windows")
                _ = os.system('python ' + order)  # Windows order
            else:
                print("Running on Linux")
                _ = os.system('python3 ' + order)  # Linux order

    if train:
        _ = Parallel(n_jobs=n_threads, verbose=10) \
            (delayed(process)(seed=seed, nat=nat) for seed in range(nr_seeds) for nat in nr_at)
        print('Simulation finished')


    if plot_final:
        if obtain_histograms:
            # Obtain histograms and save them: offline defense case
            def proc(seed, nat):
                for mode in ['online', 'offline']:
                    if mode is 'online':
                        env_id = 'attack_and_defense_online'
                        expert_path = os.path.join(os.getcwd(), 'expert_data_online', str(nat), str(seed),
                                                   'trajs_neutral.npz')
                        rew_dir = os.path.join(os.getcwd(), 'expert_data_online', str(nat), str(seed), 'reward_giver')
                        res_path = os.path.join(os.getcwd(), 'results_de_online', str(nat), str(seed), 'policy')
                    else:
                        env_id = 'attack_and_defense'
                        expert_path = os.path.join(os.getcwd(), 'expert_data', str(nat), str(seed), 'trajs_neutral.npz')
                        rew_dir = os.path.join(os.getcwd(), 'expert_data', str(nat), str(seed), 'reward_giver')
                        res_path = os.path.join(os.getcwd(), 'results_de', str(nat), str(seed), 'policy')


                    if platform.system() == 'Windows':
                        order = 'obtain_histograms.py --env_id=' + env_id + \
                                ' --seed=' + str(seed) + \
                                ' --mode=' + str(mode) + \
                                ' --nat=' + str(nat) + \
                                ' --nt=' + str(nr_ns + nat) + \
                                ' --expert_path=' + "\"" + str(expert_path) + "\"" + \
                                ' --reward_path=' + "\"" + str(rew_dir) + "\"" + \
                                ' --results_path=' + "\"" + str(res_path) + "\""
                        print("Running on Windows")
                        _ = os.system('python ' + order)  # Windows order
                    else:
                        order = 'obtain_histograms.py --env_id=' + env_id + \
                                ' --seed=' + str(seed) + \
                                ' --mode=' + str(mode) + \
                                ' --nat=' + str(nat) + \
                                ' --nt=' + str(nr_ns + nat) + \
                                ' --expert_path=' + str(expert_path) + \
                                ' --reward_path=' + str(rew_dir) + \
                                ' --results_path=' + str(res_path)
                        print("Running on Linux")
                        _ = os.system('python3 ' + order)  # Linux order
            _ = Parallel(n_jobs=n_threads, verbose=10) \
                (delayed(proc)(seed=seed, nat=nat) for seed in range(nr_seeds) for nat in nr_at)
            print('Histograms obtained')

        def filter(signal, ws):
            # Moving average filter
            signal_out = np.zeros_like(signal)
            for i in range(len(signal_out)):
                if i < ws:
                    signal_out[i] = np.mean(signal[0: i + 1])
                else:
                    signal_out[i] = np.mean(signal[i - ws: i + 1])
            return signal_out

        print('Plotting...')

        def filter_seeds(data, nr_filter, key='MtR', mode='last'):
            for i in reversed(range(len(data))):
                if data[i] is None:
                    del data[i]
            if mode == 'last':
                v = [(data[sd][-1][key]) for sd in range(len(data))]
            elif mode == 'mean':
                v = [np.mean([data[sd][i][key] for i in range(len(data[i]))]) for sd in range(len(data))]
            else:
                raise NotImplementedError
            print(v)
            to_erase = np.argsort(v)[nr_filter:]  # The best seeds from the defense point of view!
            to_erase.sort()  # Sort the indexes
            for i in range(len(to_erase) - 1, -1, -1):
                del data[to_erase[i]]
            print([(data[sd][-1][key]) for sd in range(len(data))])
            return data

        def load_data(name):
            try:
                with open(name,'rb') as handle:
                    data = pickle.load(handle)
                return data
            except:
                return None


        # Create folder to store results
        if not os.path.exists('./results'):
            os.makedirs('./results')

        for nat in nr_at:
            # Load info values: TRAINING
            info_nd = []
            info_de = []
            info_de_online = []
            ws = 10
            for seed in range(nr_seeds):
                info_nd.append(load_data(os.path.join(os.getcwd(), 'results_nd', str(nat), str(seed), 'training_results.pickle')))
                info_de.append(load_data(os.path.join(os.getcwd(), 'results_de', str(nat), str(seed), 'training_results.pickle')))
                info_de_online.append(load_data(os.path.join(os.getcwd(), 'results_de_online', str(nat), str(seed), 'training_results.pickle')))
            info_nd = filter_seeds(info_nd, nr_seeds_plot, key='MtR', mode='last')
            info_de = filter_seeds(info_de, nr_seeds_plot, key='MtR', mode='last')
            info_de_online = filter_seeds(info_de_online, nr_seeds_plot, key='MtR', mode='last')
            # Plot 1: Mean Total Reward evolution
            vals = np.zeros((trpo_iters, nr_seeds_plot, 3))  # Iter x seed x case
            for seed, it in itertools.product(range(nr_seeds_plot), range(trpo_iters)):
                vals[it, seed, 0] = info_nd[seed][it]['MtR']
                vals[it, seed, 1] = info_de[seed][it]['MtR']
                vals[it, seed, 2] = info_de_online[seed][it]['MtR']

            mean_nd = filter(np.mean(vals[:, :, 0], axis=1), ws)
            mean_de = filter(np.mean(vals[:, :, 1], axis=1), ws)
            mean_de_online = filter(np.mean(vals[:, :, 2], axis=1), ws)
            std_nd = filter(np.std(vals[:, :, 0], axis=1), ws)
            std_de = filter(np.std(vals[:, :, 1], axis=1), ws)
            std_de_online = filter(np.std(vals[:, :, 2], axis=1), ws)

            x = np.arange(trpo_iters)
            plt.plot(x, mean_nd, 'b', label='No_def')
            plt.plot(x, mean_de, 'r', label='Def')
            plt.plot(x, mean_de_online, 'k', label='Def online')
            plt.fill_between(x, mean_nd + std_nd, mean_nd - std_nd, color='b', alpha=0.3)
            plt.fill_between(x, mean_de + std_de, mean_de - std_de, color='r', alpha=0.3)
            plt.fill_between(x, mean_de_online + std_de_online, mean_de_online - std_de_online, color='k', alpha=0.3)
            plt.title('Evolution of reward, NA = ' + str(nat))
            plt.xlabel('TRPO iteration')
            plt.ylabel('R')
            plt.legend(loc='best')
            tikz_save('./results/Evolution of reward NA ' + str(nat) + '.tikz', figureheight='\\figureheight',
                      figurewidth='\\figurewidth')
            plt.savefig('./results/Evolution of reward NA ' + str(nat) + '.png', bbox_inches='tight')
            plt.show()

            # Plot 2: Evolution of proportion of discovered agents
            vals = np.zeros((trpo_iters, nr_seeds_plot, 2, 3))  # Iter x seed x agent x case
            for seed, it in itertools.product(range(nr_seeds_plot), range(trpo_iters)):
                vals[it, seed, 0, 0] = info_nd[seed][it]['AttC'] / (
                        info_nd[seed][it]['AttC'] + info_nd[seed][it]['AttNC'])
                vals[it, seed, 1, 0] = info_nd[seed][it]['NsC'] / (info_nd[seed][it]['NsC'] + info_nd[seed][it]['NsNC'])
                vals[it, seed, 0, 1] = info_de[seed][it]['AttC'] / (
                        info_de[seed][it]['AttC'] + info_de[seed][it]['AttNC'])
                vals[it, seed, 1, 1] = info_de[seed][it]['NsC'] / (info_de[seed][it]['NsC'] + info_de[seed][it]['NsNC'])
                vals[it, seed, 0, 2] = info_de_online[seed][it]['AttC'] / (
                        info_de_online[seed][it]['AttC'] + info_de_online[seed][it]['AttNC'])
                vals[it, seed, 1, 2] = info_de_online[seed][it]['NsC'] / (
                        info_de_online[seed][it]['NsC'] + info_de_online[seed][it]['NsNC'])

            mean_nd_at = filter(np.mean(vals[:, :, 0, 0], axis=1), ws)
            mean_de_at = filter(np.mean(vals[:, :, 0, 1], axis=1), ws)
            mean_de_online_at = filter(np.mean(vals[:, :, 0, 2], axis=1), ws)
            mean_nd_ns = filter(np.mean(vals[:, :, 1, 0], axis=1), ws)
            mean_de_ns = filter(np.mean(vals[:, :, 1, 1], axis=1), ws)
            mean_de_online_ns = filter(np.mean(vals[:, :, 1, 2], axis=1), ws)
            std_nd_at = filter(np.std(vals[:, :, 0, 0], axis=1), ws)
            std_de_at = filter(np.std(vals[:, :, 0, 1], axis=1), ws)
            std_de_online_at = filter(np.std(vals[:, :, 0, 2], axis=1), ws)
            std_nd_ns = filter(np.std(vals[:, :, 1, 0], axis=1), ws)
            std_de_ns = filter(np.std(vals[:, :, 1, 1], axis=1), ws)
            std_de_online_ns = filter(np.std(vals[:, :, 1, 2], axis=1), ws)

            x = np.arange(trpo_iters)
            plt.plot(x, mean_nd_at, 'b', label='ND, AS')
            plt.plot(x, mean_nd_ns, 'g', label='ND, NS')
            plt.plot(x, mean_de_at, 'r', label='DE, AS')
            plt.plot(x, mean_de_ns, 'k', label='DE, NS')
            plt.plot(x, mean_de_online_at, 'y', label='DE online, AS')
            plt.plot(x, mean_de_online_ns, 'c', label='DE online, NS')
            plt.fill_between(x, mean_nd_at + std_nd_at, mean_nd_at - std_nd_at, color='b', alpha=0.3)
            plt.fill_between(x, mean_nd_ns + std_nd_ns, mean_nd_ns - std_nd_ns, color='g', alpha=0.3)
            plt.fill_between(x, mean_de_at + std_de_at, mean_de_at - std_de_at, color='r', alpha=0.3)
            plt.fill_between(x, mean_de_ns + std_de_ns, mean_de_ns - std_de_ns, color='k', alpha=0.3)
            plt.fill_between(x, mean_de_online_at + std_de_online_at, mean_de_online_at - std_de_online_at, color='y', alpha=0.3)
            plt.fill_between(x, mean_de_online_ns + std_de_online_ns, mean_de_online_ns - std_de_online_ns, color='c', alpha=0.3)
            plt.title('Proportion of AS/NS caught, NA = ' + str(nat))
            plt.xlabel('TRPO iteration')
            plt.ylabel('% discovered')
            plt.legend(loc='best')
            tikz_save('./results/Proportion of AS NS caught NA ' + str(nat) + '.tikz', figureheight='\\figureheight',
                      figurewidth='\\figurewidth')
            plt.savefig('./results/Proportion of AS NS caught NA ' + str(nat) + '.png', bbox_inches='tight')
            plt.show()

            # Plot 3: Proportion of transmitted bits per agent type
            vals = np.zeros((trpo_iters, nr_seeds_plot, 2, 3))  # Iter x seed x agent x case
            n_as = info_nd[0][0]['AttC'] + info_nd[0][0]['AttNC'] # Number of AS
            n_ns = info_nd[0][0]['NsC'] + info_nd[0][0]['NsNC']  # Number of NS
            for seed, it in itertools.product(range(nr_seeds_plot), range(trpo_iters)):
                vals[it, seed, 0, 0] = info_nd[seed][it]['MpbtA'] / (100 * n_as)
                vals[it, seed, 1, 0] = info_nd[seed][it]['MpbtN'] / (100 *n_ns)
                vals[it, seed, 0, 1] = info_de[seed][it]['MpbtA'] / (100 * n_as)
                vals[it, seed, 1, 1] = info_de[seed][it]['MpbtN'] / (100 * n_ns)
                vals[it, seed, 0, 2] = info_de_online[seed][it]['MpbtA'] / (100 * n_as)
                vals[it, seed, 1, 2] = info_de_online[seed][it]['MpbtN'] / (100 * n_ns)

            mean_nd_at = filter(np.mean(vals[:, :, 0, 0], axis=1), ws)
            mean_de_at = filter(np.mean(vals[:, :, 0, 1], axis=1), ws)
            mean_de_online_at = filter(np.mean(vals[:, :, 0, 2], axis=1), ws)
            mean_nd_ns = filter(np.mean(vals[:, :, 1, 0], axis=1), ws)
            mean_de_ns = filter(np.mean(vals[:, :, 1, 1], axis=1), ws)
            mean_de_online_ns = filter(np.mean(vals[:, :, 1, 2], axis=1), ws)
            std_nd_at = filter(np.std(vals[:, :, 0, 0], axis=1), ws)
            std_de_at = filter(np.std(vals[:, :, 0, 1], axis=1), ws)
            std_de_online_at = filter(np.std(vals[:, :, 0, 2], axis=1), ws)
            std_nd_ns = filter(np.std(vals[:, :, 1, 0], axis=1), ws)
            std_de_ns = filter(np.std(vals[:, :, 1, 1], axis=1), ws)
            std_de_online_ns = filter(np.std(vals[:, :, 1, 2], axis=1), ws)

            x = np.arange(trpo_iters)
            plt.plot(x, mean_nd_at, 'b', label='ND, AS')
            plt.plot(x, mean_nd_ns, 'g', label='ND, NS')
            plt.plot(x, mean_de_at, 'r', label='DE, AS')
            plt.plot(x, mean_de_ns, 'k', label='DE, NS')
            plt.plot(x, mean_de_online_at, 'y', label='DE online, AS')
            plt.plot(x, mean_de_online_ns, 'c', label='DE online, NS')
            plt.fill_between(x, mean_nd_at + std_nd_at, mean_nd_at - std_nd_at, color='b', alpha=0.3)
            plt.fill_between(x, mean_nd_ns + std_nd_ns, mean_nd_ns - std_nd_ns, color='g', alpha=0.3)
            plt.fill_between(x, mean_de_at + std_de_at, mean_de_at - std_de_at, color='r', alpha=0.3)
            plt.fill_between(x, mean_de_ns + std_de_ns, mean_de_ns - std_de_ns, color='k', alpha=0.3)
            plt.fill_between(x, mean_de_online_at + std_de_online_at, mean_de_online_at - std_de_online_at, color='y',
                             alpha=0.3)
            plt.fill_between(x, mean_de_online_ns + std_de_online_ns, mean_de_online_ns - std_de_online_ns, color='c',
                             alpha=0.3)
            plt.title('Proportion of bits transmitted, NA = ' + str(nat))
            plt.xlabel('TRPO iteration')
            plt.ylabel('% bits tx')
            plt.legend(loc='best')
            tikz_save('./results/Proportion of bits transmitted NA ' + str(nat) + '.tikz', figureheight='\\figureheight',
                      figurewidth='\\figurewidth')
            plt.savefig('./results/Proportion of bits transmitted NA ' + str(nat) + '.png', bbox_inches='tight')
            plt.show()

            # Load info values: TRAINED
            info_nd = []
            info_de = []
            info_de_online = []
            ws = 10

            for seed in range(nr_seeds):
                info_nd.append(load_data(os.path.join(os.getcwd(), 'results_nd', str(nat), str(seed), 'policy_results.pickle')))
                info_de.append(load_data(os.path.join(os.getcwd(), 'results_de', str(nat), str(seed), 'policy_results.pickle')))
                info_de_online.append(load_data(os.path.join(os.getcwd(), 'results_de_online', str(nat), str(seed), 'policy_results.pickle')))
            info_nd = filter_seeds(info_nd, nr_seeds_plot, key='mean_total_rwd', mode='mean')
            info_de = filter_seeds(info_de, nr_seeds_plot, key='mean_total_rwd', mode='mean')
            info_de_online = filter_seeds(info_de_online, nr_seeds_plot, key='mean_total_rwd', mode='mean')
            print('Trained results, NA = ' + str(nat))
            # Info 1: Mean Total Reward
            n_eval = 100
            vals = np.zeros((n_eval, nr_seeds_plot, 3))  # Iter x seed x case
            for seed, it in itertools.product(range(nr_seeds_plot), range(n_eval)):
                vals[it, seed, 0] = info_nd[seed][it]['mean_total_rwd']
                vals[it, seed, 1] = info_de[seed][it]['mean_total_rwd']
                vals[it, seed, 2] = info_de_online[seed][it]['mean_total_rwd']
            print('MEAN TOTAL REWARD', '\n',
                  'ND = ', np.mean(vals[:, :, 0]), ' +- ', np.std(vals[:, :, 0]), '\n',
                  'DE = ', np.mean(vals[:, :, 1]), ' +- ', np.std(vals[:, :, 1]), '\n',
                  'DE online = ', np.mean(vals[:, :, 2]), ' +- ', np.std(vals[:, :, 2]), '\n',
                  'p-value offline = ',
                  stats.ttest_ind(vals[:, :, 0].flatten(), vals[:, :, 1].flatten(), equal_var=False)[1], '\n',
                  'p-value online = ',
                  stats.ttest_ind(vals[:, :, 0].flatten(), vals[:, :, 2].flatten(), equal_var=False)[1])
            # Info 2: Discovered agents
            vals = np.zeros((n_eval, nr_seeds_plot, 2, 3))  # Iter x seed x agent x case
            for seed, it in itertools.product(range(nr_seeds_plot), range(n_eval)):
                vals[it, seed, 0, 0] = info_nd[seed][it]['attackers_caught'] / (info_nd[seed][it]['attackers_caught'] + info_nd[seed][it]['attackers_not_caught'])
                vals[it, seed, 1, 0] = info_nd[seed][it]['ns_caught'] / (info_nd[seed][it]['ns_caught'] + info_nd[seed][it]['ns_not_caught'])
                vals[it, seed, 0, 1] = info_de[seed][it]['attackers_caught'] / (info_de[seed][it]['attackers_caught'] + info_de[seed][it]['attackers_not_caught'])
                vals[it, seed, 1, 1] = info_de[seed][it]['ns_caught'] / (info_de[seed][it]['ns_caught'] + info_de[seed][it]['ns_not_caught'])
                vals[it, seed, 0, 2] = info_de_online[seed][it]['attackers_caught'] / (
                            info_de_online[seed][it]['attackers_caught'] + info_de_online[seed][it]['attackers_not_caught'])
                vals[it, seed, 1, 2] = info_de_online[seed][it]['ns_caught'] / (
                            info_de_online[seed][it]['ns_caught'] + info_de_online[seed][it]['ns_not_caught'])
            print('AGENTS CAUGHT', '\n',
                  'ND, AS', np.mean(vals[:, :, 0, 0]), ' +- ', np.std(vals[:, :, 0, 0]), '\n',
                  'ND, GS', np.mean(vals[:, :, 1, 0]), ' +- ', np.std(vals[:, :, 1, 0]), '\n',
                  'DE, AS', np.mean(vals[:, :, 0, 1]), ' +- ', np.std(vals[:, :, 0, 1]), '\n',
                  'DE, GS', np.mean(vals[:, :, 1, 1]), ' +- ', np.std(vals[:, :, 1, 1]), '\n',
                  'DE online, AS', np.mean(vals[:, :, 0, 2]), ' +- ', np.std(vals[:, :, 0, 2]), '\n',
                  'DE online, GS', np.mean(vals[:, :, 1, 2]), ' +- ', np.std(vals[:, :, 1, 2]), '\n',
                  'p-value, AS, offline = ',
                  stats.ttest_ind(vals[:, :, 0, 0].flatten(), vals[:, :, 0, 1].flatten(), equal_var=False)[1], '\n',
                  'p-value, GS, offline = ',
                  stats.ttest_ind(vals[:, :, 1, 0].flatten(), vals[:, :, 1, 1].flatten(), equal_var=False)[1], '\n',
                  'p-value, AS, online = ',
                  stats.ttest_ind(vals[:, :, 0, 0].flatten(), vals[:, :, 0, 2].flatten(), equal_var=False)[1], '\n',
                  'p-value, GS, online = ',
                  stats.ttest_ind(vals[:, :, 1, 0].flatten(), vals[:, :, 1, 2].flatten(), equal_var=False)[1])
            # Info 3: Proportion of transmitted bits per agent type
            vals = np.zeros((n_eval, nr_seeds_plot, 2, 3))  # Iter x seed x agent x case
            n_as = info_nd[0][0]['attackers_caught'] + info_nd[0][0]['attackers_not_caught']  # Number of AS
            n_ns = info_nd[0][0]['ns_caught'] + info_nd[0][0]['ns_not_caught']  # Number of NS
            for seed, it in itertools.product(range(nr_seeds_plot), range(n_eval)):
                vals[it, seed, 0, 0] = info_nd[seed][it]['mean_prop_bits_tx_at'] / (100 * n_as)
                vals[it, seed, 1, 0] = info_nd[seed][it]['mean_prop_bits_tx_no'] / (100 * n_ns)
                vals[it, seed, 0, 1] = info_de[seed][it]['mean_prop_bits_tx_at'] / (100 * n_as)
                vals[it, seed, 1, 1] = info_de[seed][it]['mean_prop_bits_tx_no'] / (100 * n_ns)
                vals[it, seed, 0, 2] = info_de_online[seed][it]['mean_prop_bits_tx_at'] / (100 * n_as)
                vals[it, seed, 1, 2] = info_de_online[seed][it]['mean_prop_bits_tx_no'] / (100 * n_ns)
            print('PROP OF BITS TX', '\n',
                  'ND, AS', np.mean(vals[:, :, 0, 0]), ' +- ', np.std(vals[:, :, 0, 0]), '\n',
                  'ND, NS', np.mean(vals[:, :, 1, 0]), ' +- ', np.std(vals[:, :, 1, 0]), '\n',
                  'DE, AS', np.mean(vals[:, :, 0, 1]), ' +- ', np.std(vals[:, :, 0, 1]), '\n',
                  'DE, NS', np.mean(vals[:, :, 1, 1]), ' +- ', np.std(vals[:, :, 1, 1]), '\n',
                  'DE online, AS', np.mean(vals[:, :, 0, 2]), ' +- ', np.std(vals[:, :, 0, 2]), '\n',
                  'DE online, NS', np.mean(vals[:, :, 1, 2]), ' +- ', np.std(vals[:, :, 1, 2]), '\n',
                  'p-value, AS, offline = ',
                  stats.ttest_ind(vals[:, :, 0, 0].flatten(), vals[:, :, 0, 1].flatten(), equal_var=False)[1], '\n',
                  'p-value, GS, offline = ',
                  stats.ttest_ind(vals[:, :, 1, 0].flatten(), vals[:, :, 1, 1].flatten(), equal_var=False)[1], '\n',
                  'p-value, AS, online = ',
                  stats.ttest_ind(vals[:, :, 0, 0].flatten(), vals[:, :, 0, 2].flatten(), equal_var=False)[1], '\n',
                  'p-value, GS, online = ',
                  stats.ttest_ind(vals[:, :, 1, 0].flatten(), vals[:, :, 1, 2].flatten(), equal_var=False)[1])
        print('End of simulation')






