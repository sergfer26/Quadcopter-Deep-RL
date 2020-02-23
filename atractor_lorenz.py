import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
def f(xyz, t, sigma, rho, beta):
    #xyz es un vector
    x, y, z = xyz
    return [sigma * (y - x), x * (rho - z) - y,x * y - beta * z]

sigma, rho, beta = 8, 28, 8/3.0
xyz0 = [1.0, 1.0, 1.0]
t = np.linspace(0, 25, 100000)

xyzv = odeint(f, xyz0, t, args=(sigma, rho, beta))

X = xyzv[:,0]
Y = xyzv[:,1]
Z = xyzv[:,2]

fig = go.Figure(data=[go.Scatter3d(
    x=X,
    y=Y,
    z=Z,
    mode='markers',
    marker=dict(
        size=1,
        colorscale='Viridis',
        opacity=0.8
    )
)])
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
