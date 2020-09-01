import pathlib
import os
import pickle
import glob
from torch import save, load
from datetime import datetime as dt


def paths(hidden_sizes):
    hs = hidden_sizes
    name = ''
    for s in hs:
        name += '_'+str(s)
    return "saved_models/actor_"+name+".pth","DDPG/saved_models/critic_"+name+".pth"
    

def load_nets(agent, hidden_sizes):
    path1,path2 = paths(hidden_sizes)
    agent.actor.load_state_dict(load(path1))
    agent.critic.load_state_dict(load(path2))


def save_nets(agent, hidden_sizes, subpath):
    pathlib.Path('saved_models/'+ subpath).mkdir(parents=True, exist_ok=True) 
    hs = hidden_sizes
    name = ''
    for s in hs:
        name += '_'+str(s)

    save(agent.actor.state_dict(), "saved_models/"+ subpath +"/actor"+name+".pth")
    save(agent.critic.state_dict(), "saved_models/"+ subpath +"/critic"+name+".pth")


def remove_nets(subpath):
    nets = glob.glob('saved_models/'+ subpath +'/*.pth')
    for net in nets:
        os.remove(net)


def save_buffer(buffer, subpath):
    pathlib.Path('saved_buffers/'+ subpath).mkdir(parents=True, exist_ok=True)
    with open('saved_buffers/'+ subpath +'/buffer.pickle', 'wb') as handle:
        pickle.dump(buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_buffer(subpath):
    with open('saved_buffers/'+ subpath +'/buffer.pickle', 'rb') as handle:
        b = pickle.load(handle)
    return b


def remove_buffer(subpath):
    buffers = glob.glob('buffer_models/'+ subpath +'/*.pickle')
    for b in buffers:
        os.remove(b)

