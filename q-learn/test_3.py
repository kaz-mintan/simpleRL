# coding:utf-8
# http://neuro-educator.com/rl1/

import numpy as np
import random
from sequence import *
from hand_motion import *
from dummy_evaluator import *
from dizitize import *

# [2]function which determine action a(t)
def get_action(next_state):
    d_next_state = digitize_state(next_state)
    neutral, happy, surprise, angry, sad, ir = d_next_state
    #epsilon-greedy
    epsilon = 0.5
    if epsilon <= np.random.uniform(0, 1):
        #next_action = np.argmax(q_table[neutral,happy,surprise,angry,sad,ir,:])
        next_action = np.argmax(q_table)
    else:
        next_action = random.sample(xrange(60),1)
    return next_action

# [3]function which update Q-table
# q_table[neutral][happy][surprise][angry][sad][ir][action]
def update_Qtable(q_table, state, action, reward, next_state):
    d_state = digitize_state(state)
    d_next_state = digitize_state(next_state)

    neutral, happy, surprise, angry, sad, ir = d_state
    n_neutral, n_happy, n_surprise, n_angry, n_sad, n_ir = d_next_state

    gamma = 0.99
    alpha = 0.5

    q_t = q_table[neutral,happy,surprise,angry,sad,ir,action]
    #next_Max_Q=np.argmax(q_table[neutral][happy][surprise][angry][sad][ir][action])
    #next_Max_Q=max(q_table[n_neutral][n_happy][n_surprise][n_angry][n_sad][n_ir][n_action])
    next_Max_Q=max(q_table)
    #q_table[neutral][happy][surprise][angry][sad][ir][action]\
    q_table[neutral,happy,surprise,angry,sad,ir,action]\
            = (1 - alpha) * q_t +\
            alpha * (reward + gamma * next_Max_Q)

    return q_table

# [4] start main function. set parameters
t_window = 30  #number of time window
num_episodes = 20  #number of all trials

type_face = 5
type_ir = 1

num_face = 20 #5%
num_ir = 20 #5mm
num_action = 6 #10deg

#q_table = np.random.uniform(low=0, high=1, size=(num_face*5+num_ir, num_action))
q_table = np.random.uniform(low=0, high=1,\
        size=(num_face,\
            num_face,\
            num_face,\
            num_face,\
            num_face,\
            num_ir,\
            num_action))

# [5] main tourine
#state = np.zeros((num_face*5+num_ir,t_window))
state = np.zeros((type_face+type_ir,t_window))
#state_mean = np.zeros((num_face*5+num_ir,num_episodes))
state_mean = np.zeros((type_face+type_ir,num_episodes))
action = np.zeros(num_episodes)
reward = np.zeros(num_episodes)

state[:,0] = np.array([100,0,0,0,0,30])
#action[0] = np.argmax(q_table[state])#TODO not enough
action[0] = np.random.uniform(0,70)#TODO not enough

for episode in range(num_episodes):  #repeat for number of trials
    # initialize enviroment

    mode = 'heuristic'

    if episode == 0:
        #state[:,0] = np.array([100,0,0,0,0,30])
        state[:,0] = np.array([100,0,0,0,0,30])
    else:
        state[:,0] = before_state


    for t in range(1,t_window):  #roup for 1 time window
        #next_state[t] = np.hstack((get_face(),get_ir()))
        state[:,t] = np.hstack((get_face(action[episode],'happy'),get_ir(state[type_face,t-1])))

    #print('state_mean[episode]',state_mean[:,episode])
    #print('state',state)
    state_mean[:,episode]=seq2feature(state)
    # calcurate s_{t+1}, r_{t} etc based on selected/conducted action
    reward[episode] = calc_reward(state, state, t_window, mode)

    # calcurate s_{t+1} and update q-table(as q-function)
    #q_table = update_Qtable(q_table, state[:,t-1], action[t], reward[t-1], state[:,t])
    if episode>0:
        q_table = update_Qtable(q_table, state_mean[:,episode-1], action[episode-1], reward[episode-1], state_mean[:,episode])
        #TODO:the index is correct?

    # evaluate the next action a_{t+1}
    #action[t] = get_action(next_state)    # a_{t+1} 
    action[episode] = get_action(state_mean[:,episode])    # a_{t+1} 
    before_state = state[:,t_window-1]

np.savetxt('action_pwm.csv', action, delimiter=",")
np.savetxt('reward_seq.csv', rewards, delimiter=",")
