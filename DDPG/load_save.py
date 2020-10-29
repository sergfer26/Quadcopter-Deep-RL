import pathlib
import os
import pickle
import glob
from torch import save, load
from datetime import datetime as dt


def paths(hidden_sizes, path):
    hs = hidden_sizes
    name = ''
    for s in hs:
        name += '_'+str(s)
    return path + "/saved_models/actor"+ name +".pth", path + "/saved_models/critic"+ name +".pth"
    

def load_nets(agent, hidden_sizes, subpath):
    path1, path2 = paths(hidden_sizes, subpath)
    agent.actor.load_state_dict(load(path1))
    agent.critic.load_state_dict(load(path2))


def save_nets(agent, hidden_sizes, path):
    pathlib.Path(path + '/saved_models').mkdir(parents=True, exist_ok=True) 
    hs = hidden_sizes
    sizes = ''
    for s in hs:
        sizes += '_'+str(s)

    save_net(agent.actor, path +"/saved_models", "actor"+ sizes)
    save_net(agent.critic, path +"/saved_models", "critic"+ sizes)


def save_net(net, path, name):
    save(net.state_dict(), path+'/'+ name +'.pth')


def remove_nets(path):
    nets = glob.glob(path +'/saved_models/*.pth')
    for net in nets:
        os.remove(net)


def save_buffer(buffer, path):
    pathlib.Path(path +'/saved_buffers').mkdir(parents=True, exist_ok=True)
    with open(path +'/saved_buffers/buffer.pickle', 'wb') as handle:
        pickle.dump(buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_buffer(path):
    with open(path +'/saved_buffers/buffer.pickle', 'rb') as handle:
        b = pickle.load(handle)
    return b


def remove_buffer(path):
    buffers = glob.glob(path +'buffer_models/*.pickle')
    for b in buffers:
        os.remove(b)

