# coding:utf-8
import numpy as np
import math

def sim_traj(pwm, time_range):
    #rate = np.array([1,3])
    rate = np.array([1,2])
    p_inflection = time_range*rate[0]/(rate[0]+rate[1])/2
    p_back = time_range*rate[0]/(rate[0]+rate[1])
    p_back_2 = time_range*rate[1]/(rate[0]+rate[1])
    print('time_range, p_inflection, p_back', time_range, p_inflection,p_back)

    alpha = -0.7*p_inflection/pwm
    trajectory=np.zeros(time_range)
    for t in range(time_range):
        if t < p_back:
            a = alpha
            p = p_inflection
        else:
            a = -alpha*rate[0]/rate[1]
            p = p_back_2
        trajectory[t]=pwm/(1+math.exp(a*(t-p)))
    
    return trajectory


traj=sim_traj(50,200)
np.savetxt('test_traj.csv',traj,delimiter=',')
