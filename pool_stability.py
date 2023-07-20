import time
import pathlib
from lqr_stability import stability
from multiprocessing import Process
from utils import date_as_path
from send_email import send_email
from GPS.controller import DummyController
from params import STATE_NAMES, state_space

if __name__ == '__main__':
    path = 'results_gps/23_04_22_02_26/buffer/'
    save_path = 'results_gps/23_04_22_02_26/stability/'+date_as_path()+'/control/'
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    processes = list()
    sims = int(1e4)
    eps = 4e-1
    with_x0 = True
    for i in range(7):
        save_name = f'stability_{i}'
        agent = DummyController(path, f'control_{i}.npz')
        p = Process(target=stability, args=(agent,
                                            state_space,
                                            STATE_NAMES,
                                            save_path,
                                            save_name,
                                            eps,
                                            with_x0,
                                            sims,
                                            ))
        processes.append(p)
        p.start()

    ti = time.time()
    for p in processes:
        p.join()

    tf = time.time()
    total_t = tf - ti
    send_email(credentials_path='credentials.txt',
               subject='Termino de simulaciones de control: ' + save_path,
               reciever='sfernandezm97@ciencias.unam.mx',
               message=f'tiempo total: {total_t}',
               path2images=save_path
               )
