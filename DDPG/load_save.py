import pathlib
from torch import save, load
from datetime import datetime as dt


def paths(hidden_sizes):
    hs = hidden_sizes
    name = ''
    for s in hs:
        name += '_'+str(s)
    return "saved_models/quadcopter/actor_"+name+".pth","DDPG/saved_models/quadcopter/critic_"+name+".pth"
    

def load_nets(agent, hidden_sizes):
    path1,path2 = paths(hidden_sizes)
    agent.actor.load_state_dict(load(path1))
    agent.critic.load_state_dict(load(path2))


def save_nets(agent, hidden_sizes, day):
    pathlib.Path('saved_models/'+ day).mkdir(parents=True, exist_ok=True) 
    hs = hidden_sizes
    name = ''
    for s in hs:
        name += '_'+str(s)

    save(agent.actor.state_dict(), "saved_models/"+ day+"/actor"+name+".pth")
    save(agent.critic.state_dict(), "saved_models/"+ day+"/critic"+name+".pth")
