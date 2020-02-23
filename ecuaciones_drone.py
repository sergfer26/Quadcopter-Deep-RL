import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
from numpy import sin
from numpy import cos
from numpy import tan


g = 9.81
I = (4.856*10**-3, 4.856*10**-3, 8.801*10**-3)
def f(uvwpqrPTAxyz,t,b, m, l, k):
    #El primer parametro es un vector
    #W,I tambien
    u,v,w,p,q,r,psi,theta,phi,x,y,z = uvwpqrPTAxyz
    Ixx,Iyy,Izz = I
    w1,w2,w3,w4 = W
    u_punto = r*v - q*w - g*sin(theta)
    v_punto = p*w - r*u - g*cos(theta)*sin(phi)
    w_punto = q*u - p*v + g*cos(phi)*cos(theta) - (b/m)*(np.sum(np.array([w1,w2,w3,w4])**2))
    p_punto = ((l*k)/(Ixx))*(w4**2 - w2**2) - q*r*((Izz-Iyy)/(Ixx))
    q_punto = ((l*k)/(Iyy))*(w3**2 - w1**2) - p*r*((Ixx-Izz)/(Iyy))
    r_punto = (b/Izz)*(w2**2 +  w4**2 - w1**2 - w3**2)
    psi_punto = (q*sin(phi) + r*cos(phi))*(1/cos(theta))
    theta_punto = q*cos(phi) - r*sin(phi)
    phi_punto = p + (q*sin(phi) + r*cos(phi))*tan(theta)
    x_punto = u
    y_punto = v
    z_punto = w
    return [u_punto, v_punto, w_punto, p_punto, q_punto, r_punto, psi_punto, theta_punto, phi_punto, x_punto, y_punto, z_punto]


b, m, l, k = 1.140*10**-7,0.468, 0.225, 2.980*10**-6
W = (1,1,1,1)
t = np.linspace(0,11, 10000)
uvwpqrPTAxyz = [1,1,1,1,1,1,1,1,1,10,10,100]

sol = odeint(f,uvwpqrPTAxyz,t,args = (b, m, l, k))

psi= sol[:,6]
theta = sol[:,7]
phi =  sol[:,8]
X = sol[:,9]
Y = sol[:,10]
Z = sol[:,11]

fig = go.Figure(data=[go.Scatter3d(
     x=X,
     y=Y,
     z=Z,
     mode='markers',
     marker=dict(size=1,colorscale='Viridis',opacity=0.8))])
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
