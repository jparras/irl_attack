from baselines.common import zipsame, dataset
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common import colorize
from mpi4py import MPI
from collections import deque
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from contextlib import contextmanager
from baselines.common.act_wrapper import ActWrapper
import sys
from gym import spaces
import os


def traj_segment_generator(pi, env, horizon, stochastic):

    # Initialize state variables
    t = 0
    n_agents = env.nr_agents
    new = True
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []
    time_steps = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros([horizon, n_agents], 'float32')
    vpreds = np.zeros([horizon, n_agents], 'float32')
    news = np.zeros([horizon, n_agents], 'int32')
    if isinstance(env.action_space, spaces.Box):
        ac = np.vstack([env.action_space.sample() for _ in range(n_agents)])  # Used only to initialize vectors!!
        acs = np.array([ac for _ in range(horizon)])
    elif isinstance(env.action_space, spaces.Discrete):
        ac = np.array([env.action_space.sample() for _ in range(n_agents)])
        acs = np.zeros([horizon, n_agents], 'int32')  # For discrete actions
    else:
        raise NotImplementedError
    if env.def_mech.online:  # To save data for training GAIL online
        obs_def = []
        acs_def = []
        news_def = []
        dones_def = []
        ob_def = env.get_def_obs()
        agents_def = range(env.def_mech.n_agents - env.def_mech.nr_anchor, env.def_mech.n_agents)
        obs_list_def = [[] for _ in agents_def]
        acs_list_def = [[] for _ in agents_def]
        done_list_def = [[] for _ in agents_def]
        len_list_def = [[] for _ in agents_def]
        ret_list_def = [[] for _ in agents_def]
        n_traj = 0
    prevacs = acs.copy()
    time = np.zeros(horizon, 'float32')  # To store the time of acting
    # Info to be saved in the logger
    keys_to_save = ['attackers_caught', 'attackers_not_caught', 'ns_caught', 'ns_not_caught', 'mean_total_rwd',
                    'total_rwd']
    if env.attack_mode == 'phy':
        keys_to_save.extend(['phy_fc_error_rate'])
    if env.attack_mode == 'mac':
        keys_to_save.extend(['total_mac_tx', 'total_mac_col', 'total_bits_tx', 'prop_t_tx', 'mean_prop_bits_tx_at',
                             'mean_prop_bits_tx_no'])
    info_indiv = []
    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, np.vstack(ob))
        if isinstance(env.action_space, spaces.Box):
            ac = np.clip(ac, env.action_space.low, env.action_space.high)  # To ensure actions are in the right limit!
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            info_total = {}
            for key in keys_to_save:
                aux = 0
                for i in range(len(info_indiv)):
                    aux += info_indiv[i][key]
                info_total[key] = aux / len(info_indiv)
            if env.def_mech.online:  # Call GAIL
                print('Training GAIL with ', n_traj, ' trajectories stored')
                # Save expert trajectories
                np.savez(env.def_mech.expert_path, obs=np.array(obs_list_def), acs=np.array(acs_list_def),
                         done=np.array(done_list_def), lens=np.array(len_list_def),
                         ep_rets=np.array(ret_list_def))
                env.def_mech.train_gail()
                # Restart the lists to store new samples only
                obs_list_def = [[] for _ in agents_def]
                acs_list_def = [[] for _ in agents_def]
                done_list_def = [[] for _ in agents_def]
                len_list_def = [[] for _ in agents_def]
                ret_list_def = [[] for _ in agents_def]

            if isinstance(env.action_space, spaces.Box):
                yield [
                    dict(
                        ob=np.array(obs[:, na, :]),
                        rew=np.array(rews[:, na]),
                        vpred=np.array(vpreds[:, na]),
                        new=np.array(news[:, na]),
                        ac=np.array(acs[:, na, :]),
                        prevac=np.array(prevacs[:, na, :]),
                        nextvpred=vpred[na] * (1 - new),
                        ep_rets=[epr[na] for epr in ep_rets],
                        ep_lens=ep_lens,
                        time_steps=np.array(time_steps),
                        time=time,
                    ) for na in range(n_agents)
                ], info_total
            elif isinstance(env.action_space, spaces.Discrete):
                yield [
                    dict(
                        ob=np.array(obs[:, na, :]),
                        rew=np.array(rews[:, na]),
                        vpred=np.array(vpreds[:, na]),
                        new=np.array(news[:, na]),
                        ac=np.array(acs[:, na]),
                        prevac=np.array(prevacs[:, na]),
                        nextvpred=vpred[na] * (1 - new),
                        ep_rets=[epr[na] for epr in ep_rets],
                        ep_lens=ep_lens,
                        time_steps=np.array(time_steps),
                        time=time
                    ) for na in range(n_agents)
                ], info_total
            else:
                raise NotImplementedError
            _, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            time_steps = []
            info_indiv = []
        i = t % horizon
        time_steps.append(t)
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac
        if env.attack_mode == 'mac':
            time[i] = sum(env.t_counter)
        elif env.attack_mode == 'phy':
            time[i] = env.timestep
        else:
            raise RuntimeError('Environment not recognized')
        if env.def_mech.online:
            obs_def.append(ob_def)
            news_def.append(new)

        ob, rew, new, info = env.step(ac)
        rews[i] = rew
        #mask_undetected[i] = np.logical_not(env.banned[0:env.nr_agents])
        if env.def_mech.online:
            ob_def = info['def_obs']
            acs_def.append(info['ac_list'])
            dones_def.append(new)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            info_indiv.append(info)
            # env.mac_plot(False, True, None)
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
            if env.def_mech.online:
                for ag in agents_def:
                    index = agents_def.index(ag)
                    obs_list_def[index].append(np.squeeze(np.array(obs_def)[:, ag, :]))
                    acs_list_def[index].append(np.expand_dims(np.array(acs_def)[:, ag], 1))
                    ret_list_def[index].append(0)
                    done_list_def[index].append(dones_def)
                    len_list_def[index].append(ep_lens[-1])
                    n_traj += 1  # We add an agent trajectory
                # Reset lists to add new data
                obs_def = []
                acs_def = []
                news_def = []
                dones_def = []

            sys.stdout.write('\r Current horizon length = ' + str((t + 1) % horizon) + '/' + str(horizon))
            sys.stdout.flush()
        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    new = [np.append(p["new"], 0) for p in seg]  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = [np.append(p["vpred"], p["nextvpred"]) for p in seg]

    for i, p in enumerate(seg):
        T = len(p["rew"])
        p["adv"] = gaelam = np.empty(T, 'float32')
        rew = p["rew"]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - new[i][t + 1]
            delta = rew[t] + gamma * vpred[i][t + 1] * nonterminal - vpred[i][t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        p["tdlamret"] = p["adv"] + p["vpred"]

def learn(env, policy_fn, *,
          timesteps_per_batch, # what to train on
          max_kl, cg_iters,
          gamma, lam, # advantage estimation
          entcoeff=0.0,
          cg_damping=1e-2,
          vf_stepsize=3e-4,
          vf_iters =3,
          max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
          callback=None,
          save_dir=None,
          save_flag=False,
          plot_flag=False
          ):
    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)
    oldpi = policy_fn("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = entcoeff * meanent

    vferr = tf.reduce_mean(tf.square(pi.vpred - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold (advantage--> Next line)
    surrgain = tf.reduce_mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]  # Policy variables
    var_list.extend([v for v in all_var_list if v.name.split("/")[1].startswith("me")])  # Mean embedding variables
    vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]  # Value function variables
    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start + sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g * tangent) for (g, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in
                                                    zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    act_params = {
        'name': "pi",
        'ob_space': ob_space,
        'ac_space': ac_space,
    }

    pi = ActWrapper(pi, act_params)

    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)
    #pi.load('./results_de/0/policy', policy_fn)
    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards

    info_out = [] # To save the training info

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break

        if max_timesteps:
            print(colorize(str(100 * timesteps_so_far / max_timesteps) + ' % of timesteps', color='magenta'))
        elif max_episodes:
            print(colorize(str(100 * episodes_so_far / max_episodes) + ' % of episodes', color='magenta'))
        elif max_iters:
            print(colorize(str(100 * iters_so_far / max_iters) + ' % of iters', color='magenta'))
        print("********** Iteration %i ************" % iters_so_far)

        with timed("sampling"):
            seg, info = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        ob = np.concatenate([s['ob'] for s in seg], axis=0)
        ac = np.concatenate([s['ac'] for s in seg], axis=0)
        atarg = np.concatenate([s['adv'] for s in seg], axis=0)
        tdlamret = np.concatenate([s['tdlamret'] for s in seg], axis=0)
        vpredbefore = np.concatenate([s["vpred"] for s in seg], axis=0)  # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

        # if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        # if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        args = ob, ac, atarg
        fvpargs = [arr[::5] for arr in args]

        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        assign_old_eq_new()  # set old parameter values to new parameter values
        with timed("computegrad"):
            *lossbefore, g = compute_lossandgrad(*args)
        lossbefore = allmean(np.array(lossbefore))
        g = allmean(g)
        if np.allclose(g, 0):
            print("Got zero gradient. not updating")
        else:
            with timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank == 0)
            assert np.isfinite(stepdir).all()
            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)
                meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                improve = surr - surrbefore
                print("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    print("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    print("violated KL constraint. shrinking step.")
                elif improve < 0:
                    print("surrogate didn't improve. shrinking step.")
                else:
                    print("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                print("couldn't compute a good step")
                set_from_flat(thbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum()))  # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])
        info_step = {}
        info_step['loss_names'] = loss_names
        info_step['mean_losses'] = meanlosses

        #for (lossname, lossval) in zip(loss_names, meanlosses):
        #    logger.record_tabular(lossname, lossval)

        with timed("vf"):

            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((ob, tdlamret),
                                                         include_final_partial_batch=False, batch_size=64):
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        lrlocal = (seg[0]["ep_lens"], seg[0]["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))

        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        info_step["EpisodesSoFar"] = episodes_so_far
        info_step["TimestepsSoFar"] = timesteps_so_far
        info_step["TimeElapsed"] = time.time() - tstart
        # Add info values
        info_step["AttC"] =  info['attackers_caught']
        info_step["AttNC"] = info['attackers_not_caught']
        info_step["NsC"] = info['ns_caught']
        info_step["NsNC"] = info['ns_not_caught']
        info_step["MtR"] = info['mean_total_rwd']  # Mean total reward
        info_step["TtR"] = info['total_rwd']  # Total reward
        if env.attack_mode == 'phy':
            info_step["Fce"] = info['phy_fc_error_rate']
        if env.attack_mode == 'mac':
            info_step["Tmt"] = info['total_mac_tx']
            info_step["Tmc"] = info['total_mac_col']
            info_step["Tbt"] = info['total_bits_tx']
            info_step["Ptt"] = info['prop_t_tx']
            info_step["MpbtA"] = info['mean_prop_bits_tx_at']
            info_step["MpbtN"] = info['mean_prop_bits_tx_no']
        info_out.append(info_step)

        print('Iteration results : ',
              'AttC = ', info_step['AttC'],
              'AttNC = ', info_step['AttNC'],
              'NsC = ', info_step['NsC'],
              'NsNC = ', info_step['NsNC'],
              'MtR = ', info_step['MtR'],
              'TtR = ', info_step['TtR'],
              "EpThisIter", len(lens)
              )

    return pi, info_out


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]