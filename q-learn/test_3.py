# coding:utf-8
# http://neuro-educator.com/rl1/

import numpy as np
import time

# reward function
def calc_reward(state, state_predict, mode):
    # coefficient
    c = np.array([1,1,1,-1,-1]) #for delta mode
    h = np.array([1,1,1,-1,-1]) #for heuristic mode
    reward = 0

    # extract face array (must be time sequence data)
    face = state[0:5,:] #in numpy, the 5 of the 0:5 is not included
    face_post = face[1:] #for delta mode
    face_predict = state_predict[0:5,:] #for predict mode

    if mode == 'delta':
        d_face = face_post - face[:t_sample-1]
        reward = np.mean(np.dot(c,d_face),axis=1)

    elif mode == 'heuristic':
        reward = np.mean(np.dot(h,face),axis=1))

    elif mode == 'predict':
        e_face = face_predict - face
        reward = np.mean(e_face)

    return reward

def seq2feature(state):
    state_feature = np.mean(state, axis=1)
    return state_feature

# [2]function which determine action a(t)
def get_action(next_state):
    #epsilon-greedy
    epsilon = 0.5
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0, 1])
    return next_action

# [3]function which update Q-table
def update_Qtable(q_table, state, action, reward, next_state):
    gamma = 0.99
    alpha = 0.5
    next_Max_Q=np.argmax(q_table[next_state])
    q_table[state, action] = (1 - alpha) * q_table[state, action] +\
            alpha * (reward + gamma * next_Max_Q)

    return q_table

# [4] start main function. set parameters
t_window = 200  #number of time window
num_episodes = 2000  #number of all trials

num_action = 60 #deg
num_face = 100 #%
num_ir = 100 #mm

q_table = np.random.uniform(
    low=0, high=1,
    size=(num_face*5+num_ir, num_action))

#total_reward_vec = np.zeros(num_consecutive_iterations)  #contains rewards of each trial
#final_x = np.zeros((num_episodes, 1))  #contains a value of x (t=200) after learning
#islearned = 0  #flg of finishing learning
#isrender = 0  #flg of drawing

# [5] main tourine
for episode in range(num_episodes):  #repeat for number of trials
    # initialize enviroment
    state = np.zeros((num_face*5+num_ir,t_window))
    action = np.zeros(num_episodes)
    reward = np.zeros(num_episodes)

    mode = 'predict'

    state[:,0] = np.hstack((get_face(), get_ir()))#TODO#return 
    action[0] = np.argmax(q_table[state])

    for t in range(1,t_window):  #roup for 1 time window
        #next_state[t] = np.hstack((get_face(),get_ir()))
        state[:,t] = np.hstack((get_face(),get_ir()))

        # calcurate s_{t+1}, r_{t} etc based on selected/conducted action
        reward[t-1] = calc_reward(state[:,t],mode) #TODO how to calc?

        # calcurate s_{t+1} and update q-table(as q-function)
        q_table = update_Qtable(q_table, state[:,t-1], action[t], reward[t-1], state[:,t])

        # evaluate the next action a_{t+1}
        #action[t] = get_action(next_state)    # a_{t+1} 
        action[t] = get_action(state[:,t])    # a_{t+1} 

    np.savetxt('action_theta.csv', action, delimiter=",")
    np.savetxt('reward_seq.csv', rewards, delimiter=",")
