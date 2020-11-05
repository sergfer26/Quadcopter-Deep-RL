import os
import torch
import pathlib
import time
import glob
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.autograd
import torch.nn.functional as F
import torch.optim as optim

from torch import save
from tqdm import tqdm
from numpy import sin, cos, tan
from numpy.linalg import norm, solve
from matplotlib import pyplot as plt
from scipy.integrate import odeint

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, random_split

from DDPG.env.quadcopter_env import QuadcopterEnv
from DDPG.utils import NormalizedEnv, OUNoise
from DDPG.models_merge import Actor
from Linear.ecuaciones_drone import imagen2d, f, jac_f
from Linear.step import imagen_accion, step
from DDPG.env.quadcopter_env import funcion, D
import numpy as np

from tools.my_time import my_date

from DDPG.ddpg import *
from trainRL import agent_vs_linear

plt.style.use('ggplot')

now = my_date()
print('empezó:', now)
DAY = str(now['month']) +'_'+ str(now['day']) +'_'+ str(now['hr']) + str(now['min'])
PATH = 'supervisado/'+ DAY
pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)

H_SIZES = [64, 64, 64] # dimensión de capas
ACTIONS = 4
BATCH_SIZE = 32
EPOCHS = 40
P = 0.80 # división de datos
LAMBDA = 50 # regularization

env = QuadcopterEnv()
env = NormalizedEnv(env)
agent = DDPGagent(env, hidden_sizes=H_SIZES)
noise = OUNoise(env.action_space)
noise.max_sigma = 0.0; noise.min_sigma = 0.0; noise.sigma = 0.0 
un_grado = np.pi/180
env.d = 1

df = pd.read_csv('tabla_2.csv', header=None)
device = "cpu"

if torch.cuda.is_available(): 
    device = "cuda"
    agent.actor.cuda() # para usar el gpu

print(device)
torch.__version__


plt.style.use('ggplot')


def save_net(net, path, name):
    save(net.state_dict(), path+'/'+ name +'.pth')


def actions_to_lambdas(actions):
    n, _ = actions.shape
    lambdas = np.zeros((n, 4))
    for i in range(n):
        x = solve(A, actions[i, :])
        lambdas[i, :] = x
    return lambdas
    

class CSV_Dataset(Dataset):
    
    def __init__(self, dataframe):
        actions = dataframe.values[:, 0:4]
        actions_ = env._reverse_action(actions)
        x_train = dataframe.values[:, 4:]
        y_train = actions_to_lambdas(actions_) # lambdas
        self.x_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


"""## From CSV to Dataset"""

dataset = CSV_Dataset(df)
n_samples = len(df[0])

n_train = int(P * n_samples)
n_val = n_samples - n_train
train_set, val_set = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_set, shuffle=False, batch_size=BATCH_SIZE)

"""## Training"""

criterion = nn.MSELoss()
optimizer = optim.Adam(agent.actor.parameters(), lr=0.001, weight_decay=0.0005)


def training_loop(train_loader, model, optimizer, loss_function, lam=LAMBDA, valid=False):
    running_loss = 0.0
    if valid: 
        model.eval() # modo de validación del modelo 
    for i, data in enumerate(train_loader, 0):
        X, Y = data
        X = X.to(device)
        Y = Y.to(device)
        if not valid:
            optimizer.zero_grad() # reinicia el gradiente
        
        Y_hat = model(X)
        
        actions = agent.lambdas_to_action(Y); actions_hat = agent.lambdas_to_action(Y_hat)
        actions = env._action(actions); actions_hat = env._action(actions_hat)

        lam = torch.tensor(lam)
        l2_reg = torch.tensor(0.)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss = loss_function(actions, actions_hat) + lam * l2_reg
        if not valid:
            loss.backward() # cálcula las derivadas 
            optimizer.step() # paso de optimización 
            
        running_loss += loss.item()
        
        avg_loss = running_loss/(i + 1)
        
    return avg_loss


def train_model(epochs, model, optimizer, train_loader, val_loader, criterion, n_train, n_val, path=PATH, lam=LAMBDA):
    train_time = 0
    epoch_loss = []
    val_loss = []
    best = - np.inf
    for k in range(epochs):
        start_time = time.time()
        loss_train = training_loop(train_loader, model, optimizer, criterion, lam=LAMBDA, valid=False)
        train_time +=  time.time() - start_time
        loss_val = training_loop(val_loader, model, None, criterion, lam=LAMBDA, valid=True)
        epoch_loss.append(loss_train)
        val_loss.append(loss_val)
        path1 = PATH + '/sim_'+ str(k) +'.png'
        path2 = PATH + '/actions_'+ str(k) +'.png'
        paths =[path1, path2]
        r = agent_vs_linear(False, agent, env, noise, show=False, paths=paths)
        if best < r:
            best = r
            save_net(agent.actor, path, 'actor')
        else:
            imgs = glob.glob(path +'/*_'+ str(k) +'.png')
            for img in imgs:
                os.remove(img)

    print("--- %s seconds ---", train_time)
    return epoch_loss, val_loss


def plot_loss(epoch_loss, val_loss, lam=LAMBDA, show=True, path=PATH):
    plt.plot(epoch_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.title('$\lambda ={}$'.format(lam))
    if show:
        plt.show()
    else:
        plt.savefig(path+'/loss.png')
        plt.close()


lambdas = [1, 5, 10, 25, 50, 150, 500, 1000]


for lam in lambdas:
    path = PATH + '/lam_'+ str(lam)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    epoch_loss, val_loss = train_model(EPOCHS, agent.actor, optimizer, train_loader, val_loader, criterion, n_train, n_val, path=path, lam=lam)
    plot_loss(epoch_loss, val_loss, show=False, path=path)
    for name, module in agent.actor.named_children():
        print('resetting ', name)
        module.reset_parameters()

