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

from torch import save, Tensor
from tqdm import tqdm
from numpy import sin, cos, tan
from numpy.linalg import norm, solve
from matplotlib import pyplot as plt
from scipy.integrate import odeint

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.modules.loss import _Loss 

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
EPOCHS = 100
P = 0.80 # división de datos
LAMBDA = 50 # regularization
DROPOUT = False

env = QuadcopterEnv()
env = NormalizedEnv(env)
agent = DDPGagent(env, hidden_sizes=H_SIZES)
noise = OUNoise(env.action_space)
noise.max_sigma = 0.0; noise.min_sigma = 0.0; noise.sigma = 0.0 
un_grado = np.pi/180
env.d = 1

df = pd.read_csv('tabla_1.csv', header=None)
device = "cpu"

if torch.cuda.is_available(): 
    device = "cuda"
    agent.actor.cuda() # para usar el gpu

print(device)
torch.__version__


plt.style.use('ggplot')


def save_net(net, path, name):
    save(net.state_dict(), path+'/'+ name +'.pth')



class CSV_Dataset(Dataset):
    
    def __init__(self, dataframe):
        actions = dataframe.values[:, 0:4]
        y_train = env._reverse_action(actions) # acciones escaladas
        x_train = dataframe.values[:, 4:]
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

class Tanh_MaxLikelihood(_Loss):
    '''
    https://stats.stackexchange.com/questions/38225/neural-net-cost-function-for-hyperbolic-tangent-activation
    '''
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(Tanh_MaxLikelihood, self).__init__(size_average, reduce, reduction)
        self.set_scale_function()

    def set_scale_function(self, name='uniform'):
        if name == 'sigmoid':
            self.scale = nn.Sigmoid()
        elif name == 'softmax':
            self.scale = nn.Softmax()
        else:
            self.scale = lambda x: (x + 1)/2
        
        self.scale_name = name

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        n = y.shape[-1]
        vec = - self.scale(y) * torch.log(self.scale(y_hat)) - (1 - self.scale(y)) * torch.log(1 - self.scale(y_hat))
        return torch.sum(vec)/n

criterion = Tanh_MaxLikelihood()
optimizer = optim.Adam(agent.actor.parameters(), lr=0.001, weight_decay=0.0005)


def training_loop(train_loader, model, optimizer, criterion=nn.MSELoss(), lam=LAMBDA, valid=False):
    running_loss = 0.0
    if valid: 
        model.eval() # modo de validación del modelo 
    for i, data in enumerate(train_loader, 0):
        X, Y = data
        X = X.to(device)
        Y = Y.to(device)
        if not valid:
            optimizer.zero_grad() # reinicia el gradiente
        
        Y_hat = model(X) # acciones escaladas estimadas
        lam = torch.tensor(lam).to(device)
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss = criterion(Y_hat, Y) + lam * l2_reg
        if not valid:
            loss.backward() # cálcula las derivadas 
            optimizer.step() # paso de optimización 
            
        running_loss += loss.item()
        
        avg_loss = running_loss/(i + 1)
        
    return avg_loss


def train_model(epochs, model, optimizer, train_loader, val_loader, criterion, n_train, n_val, lam=LAMBDA):
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
            save_net(agent.actor, PATH, 'actor')
        else:
            imgs = glob.glob(PATH +'/*_'+ str(k) +'.png')
            for img in imgs:
                os.remove(img)

    print("--- {} seconds ---".format(train_time))
    return epoch_loss, val_loss


def plot_loss(epoch_loss, val_loss, lam=LAMBDA, show=True, path=PATH):
    plt.plot(epoch_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.title('$\lambda={}$, scale function: {}, dropout$=${} '.format(lam, criterion.scale_name, DROPOUT))
    if show:
        plt.show()
    else:
        plt.savefig(path+'/loss.png')
        plt.close()


epoch_loss, val_loss = train_model(EPOCHS, agent.actor, optimizer, train_loader, val_loader, criterion, n_train, n_val)
plot_loss(epoch_loss, val_loss, show=False, path=PATH)

