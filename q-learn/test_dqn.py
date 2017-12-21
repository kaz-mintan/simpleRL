# coding:utf-8
# http://neuro-educator.com/rl1/

import numpy as np
from sequence import *
from hand_motion import *
from dummy_evaluator import *
from dizitize import *
from neural_network import *
from datetime import datetime

def normalization(array, val_max, val_min):
    x_max = np.max(array)
    x_min = np.min(array)
    a = (val_max - val_min)/(x_max - x_min)
    b = -a*x_max + val_max
    return (a, b)

# [4] start main function. set parameters
print('1',datetime.now())
t_window = 30  #number of time window
num_episodes = 2000  #number of all trials

type_face = 5
type_ir = 1
type_action = 1

num_face = 100 #%
num_ir = 100 #5mm
num_action = 60 #%:pwm

gamma = 0.99
alpha = 0.5

epsilon = 0.1
mu = 0.9
epoch = 10000

val_max = 0.8
val_min = 0.2


print('2',datetime.now())

# [5] main tourine
state = np.zeros((type_face+type_ir,t_window))
state_mean = np.zeros((type_face+type_ir,num_episodes))
action = np.zeros((1,num_episodes))
reward = np.zeros(num_episodes)

print('3',datetime.now())
state[:,0] = np.array([100,0,0,0,0,30])
action[:,0] = np.random.uniform(0,60)#TODO not enough

possible_a = np.linspace(0,60,100)

print('4',datetime.now())
# set qfunction as nn
input_size = type_face + type_ir + type_action
output_size = 1
hidden_size = (input_size + output_size )/2

q_teacher = np.zeros((output_size,num_episodes))

print('5',datetime.now())
Q_func = Neural(input_size, hidden_size, output_size)
first_iteacher = np.random.uniform(low=0,high=1,size=(input_size,2))
first_oteacher = np.random.uniform(low=0,high=1,size=(output_size,2))

print('6',datetime.now())

Q_func.train(first_iteacher.T,first_oteacher.T,epsilon, mu, epoch)

print('7',datetime.now())
acted = action[:,0]
rewed= 0
for episode in range(num_episodes):  #repeat for number of trials
    print('epi',episode,datetime.now(),'act',acted,'rew',rewed)

    mode = 'heuristic'

    if episode == 0:
        state[:,0] = np.array([100,0,0,0,0,30])
    else:
        state[:,0] = before_state

    for t in range(1,t_window):  #roup for 1 time window
        state[:,t] = np.hstack((get_face(action[:,episode],'happy','nega'),get_ir(state[type_face,t-1])))

    # calcurate s_{t+1}, r_{t} etc based on selected/conducted action
    state_mean[:,episode]=seq2feature(state)
    reward[episode] = calc_reward(state, state, t_window, mode)

    # calcurate a_{t+1}
    p_array= np.zeros((input_size,1))
    possible_q = np.zeros(num_action)
    for i in range(num_action):
        p_array[:,0]=np.hstack((state_mean[:,episode]/num_face,possible_a[i]/num_action))
        C, possible_q[i]=Q_func.predict(p_array.T)
    #print('possible',possible_q)
    next_q=np.max(possible_q)
    action[:,episode]=np.argmax(possible_q)

    # calcurate s_{t+1} and update q-table(as q-function)
    if episode>0:
        p_array[:,0]=np.hstack((state_mean[:,episode]/num_face,possible_a[i]/num_action))
        C, present_q = Q_func.predict(p_array.T)

        q_teacher[:,episode-1] = present_q + alpha*(reward[episode-1]+gamma*(next_q-present_q[0,0]))
        print('q_',q_teacher[:,episode-1])

        t_array = np.zeros((input_size,episode-1))

        if episode > 2:
            a, b = normalization(q_teacher[:,:episode-1],val_max,val_min)

            q_normal = a*q_teacher[:,:episode-1]+b*np.ones((output_size,episode-1))
            print('q_normal',q_normal)

            t_array = np.hstack((((state_mean[:,:episode-1])/num_face).T,(action[:,:episode-1]/num_action).T))
            Q_func.train(t_array, q_normal[:,:episode-1].T, epsilon, mu, epoch)

    before_state = state[:,t_window-1]
    acted = action[:,episode]
    rewed = reward[episode]

np.savetxt('action_pwm.csv', action[0,:], fmt="%.0f", delimiter=",")
np.savetxt('reward_seq.csv', reward, fmt="%.0f", delimiter=",")
