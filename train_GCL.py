import torch
import numpy as np
import pathlib
from tqdm import tqdm
from matplotlib import pyplot as plt
from env import QuadcopterEnv
from DDPG.utils import AgentEnv
from ILQR.utils import ContinuousDynamics, DiffCostNN
from ILQR.agent import iLQRAgent
from DDPG.ddpg import DDPGagent
from Linear.agent import LinearAgent
from Linear.equations import f, W0
from Linear.constants import PARAMS_STATE
from dynamics import transform_x, transform_u
from dynamics import inv_transform_u, inv_transform_x
from GCL.cost import CostNN
from GCL.gcl import GCL
from GCL.utils import Memory
from simulation import n_rollouts, rollout, plot_rollouts
from animation import create_animation
from params import STATE_NAMES, ACTION_NAMES, REWARD_NAMES, COST_NAMES
from params import PARAMS_TRAIN_GCL as params
from utils import load_memory, date_as_path
from copy import deepcopy
import send_email
from get_report import create_report


REWARD_UPDATES = params['REWARD_UPDATES']
DEMO_SIZE = params['DEMO_SIZE']
SAMP_SIZE = params['SAMP_SIZE']
SHOW = params['SHOW']
N = params['n']
# per_sample = params['per_sample']

if not SHOW:
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def main(path):
    # 1. Creación de Ambiente
    env = QuadcopterEnv(u0=W0)
    # 1.1 Creación de Ambiente para DDPG y GCL
    env4agent = AgentEnv(deepcopy(env), tx=transform_x, inv_tx=inv_transform_x)
    # 2. Creación de agente DDPG
    agent = DDPGagent(env4agent)
    n_x, n_u = agent.num_states, agent.num_actions
    agent.memory_samp = Memory(max_size=int(
        1e6), state_dim=n_x, action_dim=n_u, T=env.steps - 1)
    # 3.1 Caraga de experiencia
    expert = LinearAgent(env)
    high = np.array(list(PARAMS_STATE.values()))
    low = -high
    env.set_obs_space(low=low, high=high)
    memory = load_memory(expert, env, n=1000, n_x=n_x,
                         t_x=transform_x, t_u=transform_u)
    print('Termino de carga de demostraciones')
    # 4. Creación de background Dynamics y costo parametrizado
    dt = env.time[-1] - env.time[-2]
    _n_x = env.observation_space.shape[0]
    dynamics = ContinuousDynamics(f, _n_x, n_u, dt=dt, u0=W0)
    # 4.1 Creación de costo parametrizado
    cost_net = CostNN(n_x, n_u)
    cost = DiffCostNN(cost_net, _n_x, n_u,
                      t_x=transform_x, t_u=transform_u,
                      u_bound=env.action_space.high)
    low = env.observation_space.low
    high = env.observation_space.high
    ilqr = iLQRAgent(dynamics, cost, env.steps-1, low, high)
    ilqr.is_stochastic = False
    expert = LinearAgent(env)
    state_init = np.zeros(expert.num_states)
    x, u, _ = rollout(expert, env, state_init=state_init)

    ilqr.fit_control(x0=state_init, us_init=u, callback=True)
    print('Termino de ajuste iLQR')
    _, _, new_states, _ = memory.sample(DEMO_SIZE)
    # ilqr.fit_dynamics(np.apply_along_axis(inv_transform_x, -1, new_states))
    # 5. Creación de instancia de método GCL
    gcl = GCL(memory, ilqr, env, tx=transform_x, tu=transform_u,
              inv_tx=inv_transform_x, inv_tu=inv_transform_u,
              x_dim=9, y_dim=9)

    loss = {x: list() for x in ['policy', 'critic', 'ioc']}
    gcl.set_cost(cost_net)
    # ilqr.is_stochastic = True

    with tqdm(total=REWARD_UPDATES) as pbar:
        for _ in range(REWARD_UPDATES):
            # states, actions, _ = nSim(
            #    ilqr,
            #    env,
            #    n=int(SAMP_SIZE * (1 - per_sample)),
            #    t_x=transform_x,
            #    t_u=transform_u
            # )
            # new_states = states[:, 1:]
            # states = states[:, :-1]
            agent.eval()
            states_policy, actions_policy, _ = n_rollouts(
                agent,
                env4agent,
                n=SAMP_SIZE,
                t_u=transform_u
            )
            new_states_policy = states_policy[:, 1:, :]
            states_policy = states_policy[:, :-1, :]
            states = states_policy  # np.vstack([states_policy, states])
            actions = actions_policy  # np.vstack([actions_policy, actions])
            # np.vstack([new_states_policy, new_states])
            new_states = new_states_policy
            dones = np.zeros((actions.shape[0], actions.shape[1]), dtype=bool)
            dones[:, -1] = True
            gcl.memory_samp.push(
                states=states,
                actions=actions,
                next_states=new_states,
                dones=dones)
            loss['ioc'].append(gcl.train_cost(
                demo_size=DEMO_SIZE, samp_size=SAMP_SIZE))
            states_tensor = states_policy.reshape(
                states_policy.shape[0] * states_policy.shape[1], -1)
            actions_tensor = actions_policy.reshape(
                actions_policy.shape[0] * actions_policy.shape[1], -1)
            new_states_tensor = new_states_policy.reshape(
                new_states_policy.shape[0] * new_states_policy.shape[1], -1)
            states_tensor = torch.FloatTensor(states_tensor)
            actions_tensor = torch.FloatTensor(actions_tensor)
            new_states_tensor = torch.FloatTensor(new_states_tensor)
            gcl.cost.eval()
            rewards_tensor = - gcl.cost(states_tensor, actions_tensor)
            agent.train()
            policy_loss, critic_loss = agent.fit(states_tensor,
                                                 actions_tensor,
                                                 rewards_tensor,
                                                 new_states_tensor)
            gcl.ilqr.fit_control(
                x0=state_init, us_init=gcl.ilqr._nominal_us, callback=False)
            loss['policy'].append(- policy_loss)
            loss['critic'].append(critic_loss)
            pbar.set_postfix(ioc='{:.2f}'.format(loss['ioc'][-1]),
                             policy='{:.2f}'.format(loss['policy'][-1]),
                             critic='{:.2f}'.format(loss['critic'][-1])
                             )
            pbar.update(1)

    plt.style.use("fivethirtyeight")
    # fig, axes = plt.subplots(2, 2, dpi=250)
    fig = plt.figure(figsize=(12, 12), dpi=250)
    gs = fig.add_gridspec(nrows=2, ncols=2)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    for ax, key in zip([ax1, ax2, ax3], loss.keys()):
        ax.plot(loss[key])
        ax.set_xlabel('episodes')
        ax.set_title(key + ' loss')
    fig.savefig(path + 'train_performance.png')
    env4agent.noise_on = False
    states, actions, scores = n_rollouts(
        agent, env4agent, n=N, t_x=inv_transform_x)
    x = np.apply_along_axis(transform_x, -1, states[:, :-1])
    u = np.apply_along_axis(transform_u, -1, actions)
    costs = - gcl.cost.to_numpy(x, u)
    cum_costs = np.cumsum(costs, axis=1)
    scores = np.concatenate([scores, costs, cum_costs], axis=-1)
    fig1, _ = plot_rollouts(states, env4agent.time, STATE_NAMES)
    fig1.savefig(path + 'state_rollouts.png')
    fig2, _ = plot_rollouts(actions, env4agent.time, ACTION_NAMES)
    fig2.savefig(path + 'action_rollouts.png')
    fig3, _ = plot_rollouts(scores, env4agent.time, REWARD_NAMES + COST_NAMES)
    fig3.savefig(path + 'score_rollouts.png')
    subpath = path + 'sample_rollouts/'
    pathlib.Path(subpath).mkdir(parents=True, exist_ok=True)
    print('Termino de simualcion...')
    create_animation(states, actions, env.time, scores=scores,
                     state_labels=STATE_NAMES,
                     action_labels=ACTION_NAMES,
                     score_labels=REWARD_NAMES + COST_NAMES,
                     path=subpath
                     )
    agent.save(path)
    gcl.save(path)
    gcl.ilqr.save(path)
    create_report(path,
                  title='Entrenamiento GCL',
                  method='gcl',
                  extra_method='ilqr'
                  )
    return path


if __name__ == "__main__":
    PATH = 'results_gcl/' + date_as_path() + '/'
    pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
    if not SHOW:
        send_email.report_sender(main, args=[PATH])
    else:
        main(PATH)
