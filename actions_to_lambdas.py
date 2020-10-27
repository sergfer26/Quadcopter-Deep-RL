import pandas as pd
import numpy as np
from numpy.linalg import solve 

A = np.array([[1, 1, 1, 1], [1, 0, -1, 0], [0, 1, 0, -1], [1, -1, 1, -1]]).T


def actions_to_lambdas(actions):
    n, _ = actions.shape
    lambdas = np.zeros((n, 4))
    for i in range(n):
        x = solve(A, actions[i, :])
        lambdas[i, :] = x
    return lambdas


df = pd.read_csv('tabla.csv', header=None)
actions = df.values[:, 0:4]
lambdas = actions_to_lambdas(actions)

new_data = np.hstack([lambdas, df.values[:, 4:]])

new_df = pd.DataFrame(new_data)

new_df.to_csv('tabla_lambdas.csv')


