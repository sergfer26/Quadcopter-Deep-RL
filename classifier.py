from numpy.random import normal
from step import simulador
from step import*


def random_position(mean_y, sd_y):
    '''
    Genera un arreglo de normales donde cada elemento i tiene 
    media mean_y[i] y varianza sd_y[i].

    param mean_y: arreglo con medias
    param sd_y: arreglo de varianzas

    regresa: arreglo de normales
    '''
    g = lambda x, y: normal(x, y)
    return g(mean_y, sd_y)
 

def prueba(Y0):
    _, _, w_0, p_0, q_0, r_0, psi_0, theta_0, phi_0, _, _, z_0 = Y0
    Y_f = simulador(Y0, 50, 1500)[-1]
    _, _, w_f, p_f, q_f, r_f, psi_f, theta_f, phi_f, _, _, z_f = Y_f
    print( np.abs(z_f-z_0) ,  np.abs(w_f))
    z = np.abs(z_f-z_0) < 0.02 and np.abs(w_f) < 0.01
    psi = np.abs(psi_f) < 0.01 and np.abs(r_f) < 0.01
    phi = np.abs(phi_f) < 0.01 and np.abs(p_f) < 0.01
    theta = np.abs(theta_f) < 0.01 and np.abs(q_f-q_0) < 0.01
    return list(map(int,[z,psi,phi,theta]))


if __name__ == "__main__":
    Y0 = np.array([0, 0, 0, 0, 0, 0, pi/20, pi/20, pi/20, 0, 0, 15])
    print(prueba(Y0))
