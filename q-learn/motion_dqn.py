# coding:utf-8
# http://neuro-educator.com/rl1/

import numpy as np
from sequence import *
from hand_motion import *
from dummy_evaluator import *
from neural_network import *
from datetime import datetime

import sys

select_episode = 10

t_window = 30  #number of time window
num_episodes = 200  #number of all trials

type_face = 5
type_ir = 1
type_action = 2

num_face = 100 #%
num_ir = 100 #5mm
num_action = np.array([40,50]) #%:pwm

gamma = 0.9
alpha = 0.5

epsilon = 0.1
mu = 0.9
epoch = 1000

val_max = 0.8
val_min = 0.2

def normalization(array, val_max, val_min):
    x_max = np.max(array)
    x_min = np.min(array)
    a = (val_max - val_min)/(x_max - x_min)
    b = -a*x_max + val_max
    return (a, b)

def inv_normalization(a, b, norm_q):
    return (norm_q - b)/a

def volts(q_teacher, q_k, T=1):
    exp_1=np.sum(np.exp(q_teacher/T))
    exp_2=np.exp(q_k/T)
    return exp_2/exp_1

def select_teach(input_array, q_teacher,episode,num=select_episode):
    index_array = np.argsort(q_teacher)[::-1]
    selected_input = input_array[index_array]
    selected_output = np.sort(q_teacher)[::-1]

    return selected_input[0,:num,:], selected_output[:,:num]

#5 [4] start main function. set parameters
argvs = sys.argv
target_type = argvs[1]
target_direct = argvs[2]
mode = argvs[3]

# [5] main tourine
state = np.zeros((type_face+type_ir,t_window))
state_before = np.zeros_like(state)
state_predict = np.zeros_like(state)
state_mean = np.zeros((type_face+type_ir,num_episodes))

action = np.zeros((type_action,num_episodes))
reward = np.zeros(num_episodes)
random = np.zeros(num_episodes)
face_predict = np.zeros((1,type_face))

state[:,0] = np.array([100,0,0,0,0,30])
action[:,0] = np.random.uniform(low=0,high=1,size=(1,type_action))

possible_a = np.tile(np.linspace(0,1,100),(type_action,1))

# possible_a をタイルにして、type_action x 100の行列にする

## set qfunction as nn
q_input_size = type_face + type_ir + type_action
q_output_size = 1
q_hidden_size = (q_input_size + q_output_size )/3

q_teacher = np.zeros((q_output_size,num_episodes))

Q_func = Neural(q_input_size, q_hidden_size, q_output_size, epsilon, mu, epoch)
q_first_iteacher = np.random.uniform(low=0,high=1,size=(q_input_size,2))
q_first_oteacher = np.random.uniform(low=0,high=1,size=(q_output_size,2))

Q_func.train(q_first_iteacher.T,q_first_oteacher.T)

if mode == 'predict':
    ## set predict function as nn
    p_input_size = type_face + type_ir + type_action
    p_output_size = type_face
    p_hidden_size = (q_input_size + q_output_size )/3

    p_teacher = np.zeros((p_output_size,num_episodes))

    P_func = Neural(p_input_size, p_hidden_size, p_output_size, epsilon, mu, epoch)
    p_first_iteacher = np.random.uniform(low=0,high=1,size=(p_input_size,2))
    p_first_oteacher = np.random.uniform(low=0,high=1,size=(p_output_size,2))

    P_func.train(p_first_iteacher.T,p_first_oteacher.T)

rewed= 0.0
acted = action[:,0]

# main loop
for episode in range(num_episodes-1):  #repeat for number of trials
    state = np.zeros_like(state_before)
    acted = action[:,episode]
    print('epi',episode,target_type,target_direct,mode,'act',acted,'rew',rewed)

    for t in range(1,t_window):  #roup for 1 time window
        #state[:,t] = get_sensor()
        state[:,t] = np.hstack((get_face(action[:,episode],argvs[1],argvs[2],t,t_window),get_ir(state[type_face,t-1])))

    ### calcurate s_{t+1} based on the value of sensors
    state_mean[:,episode+1]=seq2feature(state)

    ### calcurate r_{t}
    reward[episode] = calc_reward(state/num_face, state_predict/num_face,
            state_before/num_face,t_window, mode)

    ### calcurate a_{t+1} based on s_{t+1}
    random[episode+1], action[:,episode+1],next_q = Q_func.gen_action(possible_a,
            num_action,num_face, state_mean, episode, type_action)

    q_teacher = Q_func.update(state_mean,num_action, num_face,action,episode,q_teacher,reward,next_q, select_episode, gamma, alpha)

    if mode == 'predict':
        state_predict, p_teacher = P_func.predict_update(state_mean, 
                state_predict, num_action,
                num_face,action, episode,p_teacher,reward,next_q,
                select_episode, gamma, alpha)

    before_state = state[:,t_window-1]
    acted = action[:,episode+1]
    rewed = reward[episode]
    state_before = state
    #print('time',action[1,episode+1])

np.savetxt('action_pwm.csv', action[0,:], fmt="%.0f", delimiter=",")
np.savetxt('reward_seq.csv', reward, fmt="%.5f",delimiter=",")
np.savetxt('situation.csv', state_mean.T,fmt="%.2f", delimiter=",")
np.savetxt('random_counter.csv', random,fmt="%.0f", delimiter=",")
