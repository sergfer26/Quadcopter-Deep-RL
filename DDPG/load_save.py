from torch import save, load

PATH1 = "DDPG/saved_models/actor_64_64.pth"
PATH2 = "DDPG/saved_models/critic_64_64.pth"

def load_nets(agent, path1=PATH1, path2=PATH2):
    agent.actor.load_state_dict(load(path1))
    agent.critic.load_state_dict(load(path2))


def save_nets(agent, hidden_sizes):
    hs = hidden_sizes
    name = ''
    for s in hs:
        name += '_'+str(s)
    save(agent.actor.state_dict(), "DDPG/saved_models/actor_"+name+".pth")
    save(agent.critic.state_dict(), "DDPG/saved_models/critic_"+name+".pth")