'''
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
'''

import argparse
import os.path as osp
import logging
from mpi4py import MPI

import numpy as np
import gym

from baselines.gail import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines import logger
from baselines.gail.dataset.mujoco_dset import Mujoco_Dset
from baselines.gail.adversary import TransitionClassifier

from generate_env import generate_env
import os
import tensorflow as tf


def argsparser():
    parser = argparse.ArgumentParser("GAIL based classifier")
    parser.add_argument('--env_id', help='environment ID', default='attack_for_gail')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--nat', help='AS', type=int, default=1)
    parser.add_argument('--ns', help='NS', type=int, default=10)
    parser.add_argument('--expert_path', type=str, default='./expert_data/0/trajs_neutral.npz')
    parser.add_argument('--attacker_path', type=str, default='no')  # Only for plotting ('no' avoids plotting)
    parser.add_argument('--reward_giver_path', type=str, default='./expert_data/0/reward_giver')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='./gail_model')
    parser.add_argument('--log_dir', help='the directory to save log file', default='./log')
    parser.add_argument('--results_dir', help='the directory to save results', default='./results')
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=3)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=256)
    parser.add_argument('--adversary_hidden_size', type=int, default=256)
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=300)
    parser.add_argument('--num_iters', help='number of training iters', type=int, default=10)
    # Behavior Cloning
    boolean_flag(parser, 'pretrained', default=False, help='Use BC to pretrain')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e4)
    return parser.parse_args()


def get_task_name(args):
    task_name = args.algo + "_gail."
    if args.pretrained:
        task_name += "with_pretrained."
    if args.traj_limitation != np.inf:
        task_name += "transition_limitation_%d." % args.traj_limitation
    task_name += args.env_id.split("-")[0]
    task_name = task_name + ".g_step_" + str(args.g_step) + ".d_step_" + str(args.d_step) + \
        ".policy_entcoeff_" + str(args.policy_entcoeff) + ".adversary_entcoeff_" + str(args.adversary_entcoeff)
    task_name += ".seed_" + str(args.seed)
    return task_name


def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    ge = generate_env(args.env_id, na=args.nat, nt=args.nat + args.ns)
    env = ge.generate()

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)

    dataset_neutral = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation,
                                  n_agents=args.nat + args.ns)
    if args.attacker_path == 'no':
        dataset_attack = None
    else:
        dataset_attack = Mujoco_Dset(expert_path=args.attacker_path, traj_limitation=args.traj_limitation)
    reward_giver = TransitionClassifier(env.observation_space_def, env.action_space,
                                        args.adversary_hidden_size, entcoeff=args.adversary_entcoeff)

    train(env,
          args.seed,
          policy_fn,
          reward_giver,
          dataset_neutral,
          dataset_attack,
          args.algo,
          args.g_step,
          args.d_step,
          args.policy_entcoeff,
          args.num_iters,
          args.save_per_iter,
          args.checkpoint_dir,
          args.log_dir,
          args.pretrained,
          args.BC_max_iter,
          args.results_dir,
          args.reward_giver_path,
          args.nat + args.ns,
          task_name
          )
    env.close()


def train(env, seed, policy_fn, reward_giver, dataset_neutral, dataset_attack, algo,
          g_step, d_step, policy_entcoeff, num_iters, save_per_iter,
          checkpoint_dir, log_dir, pretrained, BC_max_iter, results_dir, reward_save_name, n_agents, task_name=None):

    if pretrained:
        # Pretrain with behavior cloning
        from baselines.gail import behavior_clone
        pretrained_weight = behavior_clone.learn(env, policy_fn, dataset_neutral, max_iters=BC_max_iter)
    # Check if there exists a policy: in that case, load it
    #if os.path.isfile(reward_save_name + '.index'):
        '''
        U.initialize()
        saver = tf.train.Saver(var_list=policy_fn.get_variables())  # Load only variables under the adversary scope
        saver.restore(tf.get_default_session(), reward_save_name)
        U.save_variables(savedir_fname, variables=pi.get_variables())
        return savedir_fname
        '''
    #    pretrained_weight = reward_save_name
    #else:
    pretrained_weight = None

    if algo == 'trpo':
        from baselines.gail import trpo_mpi
        # Set up for MPI seed
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)
        pi, reward_giver, info = trpo_mpi.learn(env, policy_fn, reward_giver, dataset_neutral, rank,
                                                pretrained=pretrained, pretrained_weight=pretrained_weight,
                                                g_step=g_step, d_step=d_step,
                                                entcoeff=policy_entcoeff,
                                                max_iters=num_iters,
                                                ckpt_dir=checkpoint_dir, log_dir=log_dir,
                                                save_per_iter=save_per_iter,
                                                timesteps_per_batch=2500,
                                                max_kl=0.01, cg_iters=10, cg_damping=0.1,
                                                gamma=0.995, lam=0.98,
                                                vf_iters=5, vf_stepsize=1e-3,
                                                task_name=task_name)
        reward_giver.save_model(reward_save_name)  # Save trained reward giver model
        '''
        def obtain_reward_vector(dataset, reward_giver, n_agents):
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

        rew_neutral = obtain_reward_vector(dataset_neutral, reward_giver, n_agents)
        if dataset_attack is not None:
            rew_attack = obtain_reward_vector(dataset_attack, reward_giver)
        import matplotlib.pyplot as plt
        nbins = 100
        if dataset_attack is not None:
            bins = np.linspace(min([np.amin(rew_attack), np.amin(rew_neutral)]),
                               max([np.amax(rew_attack), np.amax(rew_neutral)]), nbins)
        else:
            bins = np.linspace(np.amin(rew_neutral), np.amax(rew_neutral), nbins)
        if n_agents == 0:
            plt.hist(rew_neutral, bins, alpha=0.5, label='neutral')
            if dataset_attack is not None:
                plt.hist(rew_attack, bins, alpha=0.5, label='attack')
        else:
            for ag in range(n_agents):
                plt.hist(rew_neutral[ag], bins, alpha=0.5, label='neutral_' + str(ag))
                if dataset_attack is not None:
                    plt.hist(rew_attack[ag], bins, alpha=0.5, label='attack_' + str(ag))
            plt.legend(loc='best')
            plt.title('Histogram per agent')
            plt.show()

            plt.hist(rew_neutral.flatten(), bins, alpha=0.5, label='neutral')
            if dataset_attack is not None:
                plt.hist(rew_attack.flatten(), bins, alpha=0.5, label='attack')
            plt.legend(loc='best')
            plt.title('Histogram total')
            plt.show()
        '''

        #from eval_policy import PolicyEvaluator
        #PolicyEvaluator(env, pi, results_dir, mode='act', info=info)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    args = argsparser()
    main(args)