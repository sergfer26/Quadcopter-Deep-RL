from DDPG.models import Actor
import pandas as pd
import numpy as np
from numpy import sin, cos, tan
from numpy.linalg import norm
import time
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from Linear.ecuaciones_drone import imagen2d
# from torchsummary import summary

h_sizes = [18, 64, 64, 64]

net = Actor(h_sizes, 4)

G = 9.81
I = (4.856*10**-3, 4.856*10**-3, 8.801*10**-3)
B, M, L = 1.140*10**(-6), 1.433, 0.225
K = 0.001219  # kt

sec = lambda x: 1/cos(x)



def f(y, t, w1, w2, w3, w4):
    #El primer parametro es un vector
    #W,I tambien
    u, v, w, x, y, z, p, q, r, psi, theta, phi = y
    Ixx, Iyy, Izz = I
    W = np.array([w1, w2, w3, w4])
    du = r * v - q * w - G * sin(theta)
    dv = p * w - r * u - G * cos(theta) * sin(phi)
    dw = q * u - p * v + G * cos(phi) * cos(theta) - (K/M) * norm(W) ** 2
    dp = ((L * B) / Ixx) * (w4 ** 2 - w2 ** 2) - q * r * ((Izz - Iyy) / Ixx)
    dq = ((L * B) / Iyy) * (w3 ** 2 - w1 ** 2) - p * r * ((Ixx - Izz) / Iyy)
    dr = (B / Izz) * (w2 ** 2 + w4 ** 2 - w1 ** 2 - w3 ** 2)
    dpsi = (q * sin(phi) + r * cos(phi)) * (1 / cos(theta))
    dtheta = q * cos(phi) - r * sin(phi)
    dphi = p + (q * sin(phi) + r * cos(phi)) * tan(theta)
    dx = u
    dy = v
    dz = w
    return du, dv, dw, dx, dy, dz, dp, dq, dr, dpsi, dtheta, dphi

def D(angulos):
    '''
        Obtine la matriz de rotación
    '''
    z, y, x = angulos # psi, theta, phi
    R = np.array([
        [cos(z) * cos(y), cos(z) * sin(y) * sin(x) - sin(z) * cos(x), 
        cos(z) * sin(y) * cos(x) + sin(z) * sin(x)],
        [sin(z) * cos(y), sin(z) * cos(y) * sin(x) + cos(z) * cos(x), 
        sin(z) * sin(y) * cos(x) - cos(z) * sin(x)], 
        [- sin(y), cos(y) * sin(x), cos(y) * cos(x)]
    ])
    return R


def funcion(state):
    '''
        Obtiene un estado de 18 posiciones
    '''
    angulos = state[9:]
    state = state[0:9]
    orientacion = np.matrix.flatten(D(angulos))
    return np.concatenate([state, orientacion])

def step(W, y, t, jac=None):
    '''
    Obtiene una solución numérica de dy=f(y) para un tiempo t+1

    param W: arreglo de velocidades de los  4 rotores
    param y: arreglo de 12 posiciones del cuadricoptero
    param t: un intervalo de tiempo

    regresa: y para el siguiente paso de tiempo
    '''
    #import pdb; pdb.set_trace()
    w1, w2, w3, w4 = W
    return odeint(f, y, t, args=(w1, w2, w3, w4) ,Dfun=jac)

def read_data(file_path): 
    dataframe = pd.read_csv(file_path, header=None)
    return dataframe

min_vals = lambda x, df: df[x].min()
max_vals = lambda x, df: df[x].max()

def normalize(x, mx, mn):
    return (x - mx) / (mx - mn)

def inv_normalize(x, mx, mn):
    return x * (mx - mn) + mn

df = read_data('tabla.csv')

'''
pd.options.mode.chained_assignment = None  # default='warn'

min_states = np.array(min_vals(range(4, 13), df))
max_states = np.array(max_vals(range(4, 13), df))
min_actions = np.array(min_vals([0, 1, 2, 3], df))
max_actions = np.array(max_vals([0, 1, 2, 3], df))

mins = np.concatenate((min_actions, min_states))
maxs = np.concatenate((max_actions, max_states))

for k in range(13):
    df[k] = normalize(df[k], maxs[k], mins[k])
'''

msk =  np.random.rand(len(df)) < 0.80

train_df = df[msk].to_numpy()
val_df = df[~msk].to_numpy()

y_train = train_df[:, 0:4]
x_train = train_df[:, 4:]
n_train = x_train.shape[0]

x_train_tensor = torch.Tensor(x_train) # transform to torch tensor
y_train_tensor = torch.Tensor(y_train)


y_val = val_df[:, 0:4]
x_val = val_df[:, 4:]
n_val = x_val.shape[0]

x_val_tensor = torch.Tensor(x_val) # transform to torch tensor
y_val_tensor = torch.Tensor(y_val)

train_set = TensorDataset(x_train_tensor, y_train_tensor)
#test_set = TensorDataset(x_test_tensor, y_test_tensor)
val_set = TensorDataset(x_val_tensor, y_val_tensor)


train_loader = DataLoader(train_set, shuffle=True, batch_size=32)
val_loader = DataLoader(val_set, shuffle=False, batch_size=32)

### Entrenamiento ###

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0005)

def training_loop(epochs, train_loader, model, optimizer, loss_function, pbar, valid=False):
    running_loss = 0.0
    if valid: 
        model.eval() # modo de validación del modelo 
    for i, data in enumerate(train_loader, 0):
        X, Y = data
    
        if not valid:
            optimizer.zero_grad() # reinicia el gradiente
        
        pred = model.forward(X)
        # Y = Y.type(torch.LongTensor) # https://stackoverflow.com/questions/60440292/runtimeerror-expected-scalar-type-long-but-found-float
        loss = loss_function(pred, Y)
        if not valid:
            loss.backward() # cálcula las derivadas 
            optimizer.step() # paso de optimización 
            
        running_loss += loss.item()
        
        avg_loss = running_loss/(i + 1)
        
        pbar.set_postfix(avg_loss='{:.4f}'.format(avg_loss))
        #import pdb; pdb.set_trace()
        pbar.update(Y.shape[0])
        
    return avg_loss


def train_model(epochs, model, optimizer, train_loader=train_loader, 
                val_loader=val_loader, criterion=criterion,
               n_train=n_train, n_val=n_val):
    train_time = 0
    epoch_loss = []
    val_loss = []

    for epoch in range(epochs):
        start_time = time.time()
        with tqdm(total = n_train, position=0) as pbar_train:
            pbar_train.set_description(f'Epoch {epoch + 1}/'+str(epochs)+' - training')
            pbar_train.set_postfix(avg_loss='0.0')
            loss_train = training_loop(epochs, train_loader, model, optimizer, 
                                                  criterion, pbar_train, valid=False)
            train_time +=  time.time() - start_time
        with tqdm(total = n_val, position=0) as pbar_val:
            pbar_val.set_description(f'Epoch {epoch +1}/'+str(epochs)+' - validation')
            pbar_val.set_postfix(avg_loss='0.0')
            loss_val = training_loop(epochs, val_loader, model, None, 
                                              criterion, pbar_val, valid=True)
    
        epoch_loss.append(loss_train)
        val_loss.append(loss_val)

    print("--- %s minutes ---", train_time)
    return epoch_loss, val_loss

epoch_loss, val_loss = train_model(50, net, optimizer, criterion=criterion)

plt.plot(epoch_loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


def simulador(Y, T, tam, jac=None):
    '''
    Soluciona el sistema de EDO usando controles en el
    intervalo [0, T].

    param Y: arreglo de la condición inicial del sistema
    param Ze: arreglo de las posiciones estables de los 4 controles
    param T: tiempo final
    param tam: número de elementos de la partición de [0, T]

    regresa; arreglo de la posición final
    '''
    Ze = (15, 0, 0, 0)
    z_e, psi_e, phi_e, theta_e = Ze
    # W0 = np.array([1, 1, 1, 1]).reshape((4, 1)) * omega_0
    X = np.zeros((tam, 12))
    X[0] = Y
    t = np.linspace(0, T, tam)
    acciones = []
    for i in range(len(t)-1):
        Y_t = funcion(Y)
        '''
        for k in range(9): # los demás ya estan en [0, 1]
            Y_t[k] = normalize(Y_t[k], max_states[k], min_states[k])
        '''

        state = Variable(torch.from_numpy(Y_t).float().unsqueeze(0))
        action = net.forward(state)
        action = action.detach().numpy()[0,:]
        
        '''
        for j in range(len(action)):
            action[j] = inv_normalize(action[j], max_actions[j], min_actions[j])
        '''

        acciones.append(action)
        Y = step(action, Y, [t[i], t[i+1]])[1]
        X[i+1] = Y
    return X, acciones

Y = np.zeros(12); Y[5] = 15
X, acciones = simulador(Y, 30, 800)

