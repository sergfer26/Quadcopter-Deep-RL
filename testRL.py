from trainRL import *
from time import time
from tqdm import tqdm
from progress.bar import Bar, ChargingBar

plt.style.use('ggplot')

if len(sys.argv) == 1:
    hidden_sizes = [64, 64, 64]
else:
    hidden_sizes = sys.argv[1:]
    hidden_sizes = [int(i) for i in hidden_sizes]


env = QuadcopterEnv()
env = NormalizedEnv(env)   
agent = DDPGagent(env, hidden_sizes=hidden_sizes)
noise = OUNoise(env.action_space)
un_grado = np.pi/180
env.d = 1


def sim(flag):
    Sim(flag, agent, env, noise, show=True)


def nsim(flag, n):
    bar = Bar('Procesando:', max=n)
    nSim(flag, agent, env, noise, n, bar=bar, show=True)


def train(agent, env, noise, episodes):
    train_time = 0.0
    for episode in range(episodes):
        start_time = time()
        with tqdm(total=env.tam, position=0) as pbar_train:
            pbar_train.set_description(f'Ep {episode + 1}/'+str(episodes)) 
            _, _, reward = training_loop(agent, env, noise, pbar_train)
            train_time +=  time() - start_time
    print(train_time)


p0 = np.zeros(12)
p1 = np.zeros(12); p1[3:6] = 0.5; p1[9:] = 0.5 * un_grado
p2 = np.zeros(12); p2[3:6] = 1; p2[9:] = 1 * un_grado
p3 = np.zeros(12); p3[3:6] = 1.5; p3[9:] = 1.5 * un_grado
p4 = np.zeros(12); p4[3:6] = 2; p4[9:] = 2 * un_grado
p5 = np.zeros(12); p5[3:6] = 2.5; p5[9:] = 2.5 * un_grado
p6 = np.zeros(12); p6[3:6] = 3; p6[9:] = 3 * un_grado
p7 = np.zeros(12); p7[3:6] = 3.5; p7[9:] = 3.5 * un_grado
p8 = np.zeros(12); p8[3:6] = 4; p8[9:] = 4 * un_grado
p9 = np.zeros(12); p9[3:6] = 4.5; p9[9:] = 4.5 * un_grado
p10 = np.zeros(12); p10[3:6] = 5; p10[9:] = 5 * un_grado


P = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
E = [500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

env.flag = False

for p, e in zip(P, E):
    env.p = p
    train(agent, env, noise, int(e))
    env.set_time(96000, 3600)
    nsim(False, 1)
    env.flag = False
    env.set_time(800, 30)
    agent.memory.remove()


env.set_time(96000, 3600)
sim(True)
env.set_time(800, 30)