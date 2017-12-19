# coding:utf-8
import numpy as np
import math

def pwm2theta(pwm):
    thre_a= 25
    thre_b= 50
    if pwm < thre_a:
        theta = 0
    elif pwm < thre_b:
        theta = pwm - 25.0
    else:
        theta = pwm * 1.5 - 50.0
    return theta

def sim_traj(pwm, time_range):
    rate = np.array([1,2])

    mu = time_range*rate[0]/(rate[0]+rate[1])
    sig = mu/3.0
    print('mu,sig',mu,sig)

    trajectory=np.zeros(time_range)
    for t in range(time_range):
        if t < mu:
            sigma = sig
        else:
            sigma = sig*rate[1]
        trajectory[t]=pwm2theta(pwm)*math.exp(-pow((t-mu),2)/(2*pow(sigma,2)))
        print(t,trajectory[t])
    return trajectory

traj=sim_traj(50,30)
np.savetxt('test_traj.csv',traj,delimiter=',')
