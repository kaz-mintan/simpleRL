# coding:utf-8
# http://neuro-educator.com/rl1/

import numpy as np
from sequence import *
from hand_motion import *
from dummy_evaluator import *
from dizitize import *
from neural_network import *

# [4] start main function. set parameters
t_window = 30  #number of time window
num_episodes = 2  #number of all trials

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


# [5] main tourine
state = np.zeros((type_face+type_ir,t_window))
state_mean = np.zeros((type_face+type_ir,num_episodes))
action = np.zeros((1,num_episodes))
reward = np.zeros(num_episodes)

state[:,0] = np.array([100,0,0,0,0,30])
action[:,0] = np.random.uniform(0,60)#TODO not enough

possible_a = np.linspace(0,60,100)

# set qfunction as nn
input_size = type_face + type_ir + type_action
output_size = 1
hidden_size = (input_size + output_size )/2

q_teacher = np.zeros((output_size,num_episodes))

Q_func = Neural(input_size, hidden_size, output_size)
first_iteacher = np.zeros((input_size,2))
first_oteacher = np.zeros((output_size,2))
Q_func.train(first_iteacher.T,first_oteacher.T,epsilon, mu, epoch)

for episode in range(num_episodes):  #repeat for number of trials
    print('episode',episode)

    mode = 'heuristic'

    if episode == 0:
        state[:,0] = np.array([100,0,0,0,0,30])
    else:
        state[:,0] = before_state

    for t in range(1,t_window):  #roup for 1 time window
        state[:,t] = np.hstack((get_face(action[:,episode],'happy'),get_ir(state[type_face,t-1])))

    # calcurate s_{t+1}, r_{t} etc based on selected/conducted action
    state_mean[:,episode]=seq2feature(state)
    reward[episode] = calc_reward(state, state, t_window, mode)

    # calcurate a_{t+1}
    p_array= np.zeros((input_size,1))
    possible_q = np.zeros(num_action)
    for i in range(num_action):
        p_array[:,0]=np.hstack((state_mean[:,episode],possible_a[i]))
        C, possible_q[i]=Q_func.predict(p_array.T)
    next_q=max(possible_q)
    action[:,episode]=np.argmax(possible_q)

    # calcurate s_{t+1} and update q-table(as q-function)
    if episode>0:
        p_array[:,0]=np.hstack((state_mean[:,episode],possible_a[i]))
        C, present_q = Q_func.predict(p_array.T)

        q_teacher[:,episode-1]= present_q + alpha*(reward[episode-1]+gamma*(next_q-present_q[0,0]))
        t_array = np.zeros((input_size,episode-1))

        if episode > 2:
            t_array = np.hstack((state_mean[:,:episode-1].T,action[:,:episode-1].T))
            Q_func.train(t_array, q_teacher[:,:episode-1].T, epsilon, mu, epoch)

    before_state = state[:,t_window-1]

np.savetxt('action_pwm.csv', action[0,:], fmt="%.0f", delimiter=",")
np.savetxt('reward_seq.csv', reward, fmt="%.0f", delimiter=",")
