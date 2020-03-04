import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
from numpy import sin
from numpy import cos
from numpy import tan


g = 9.81
I = (4.856*10**-3, 4.856*10**-3, 8.801*10**-3)

def f(y, t , b, m, l, kt, W):
    #El primer parametro es un vector
    #W,I tambien
    u, v, w, p, q, r, psi, theta, phi, x, y, z = y
    Ixx, Iyy, Izz = I
    w1, w2, w3, w4 = W
    du = r*v - q*w - g*sin(theta)
    dv = p*w - r*u - g*cos(theta)*sin(phi)
    dw = q*u - p*v + g*cos(phi)*cos(theta) - (kt/m)*np.linalg.norm(W)**2
    dp = ((l*b)/(Ixx))*(w4**2 - w2**2) - q*r*((Izz-Iyy)/(Ixx))
    dq = ((l*b)/(Iyy))*(w3**2 - w1**2) - p*r*((Ixx-Izz)/(Iyy))
    dr = (b/Izz)*(w2**2 + w4**2 - w1**2 - w3**2)
    dpsi = (q*sin(phi) + r*cos(phi))*(1/cos(theta))
    dtheta = q*cos(phi) - r*sin(phi)
    dphi = p + (q*sin(phi) + r*cos(phi))*tan(theta)
    dx = u
    dy = v
    dz = w
    return du, dv, dw, dp, dq, dr, dpsi, dtheta, dphi, dx, dy, dz

kt=0.001219

b, m, l= 1.140*10**(-6), 1.433, 0.225
W2 = (1, 158, 158, 158)
W = (53.6666, 53.66, 53.6666, 53.668)
t = np.linspace(0, 11, 10000)
y = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10

sol = odeint(f, y, t, args=(b, m, l, kt, W))
sol2 = odeint(f, y, t, args=(b, m, l, kt, W2))

print(np.linalg.norm(sol-sol2))


psi = sol[:,6]
theta = sol[:,7]
phi = sol[:,8]
X = sol[:,9]
Y = sol[:,10]
Z = sol[:,11]

fig = go.Figure(data=[go.Scatter3d(
     x=X,
     y=Y,
     z=Z,
     mode='markers',
     marker=dict(size=1, colorscale='Viridis', opacity=0.8))])
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()

print(Z)
