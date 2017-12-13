# coding:utf-8
# http://neuro-educator.com/rl1/

import numpy as np
import time
from dummy import *
from q_function import *
from reward_function import *


class Q_learning():

# [2]function which determine action a(t)
def get_action(q_table, next_state, episode, action):
    #epsilon-greedy
    epsilon = 0.5 * (1 / (episode + 1))
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        #next_action = np.random.choice([0, 1])
        next_action = randint(action.size)
    return next_action

# [3]function which update Q-table
#def update_Qtable(q_table, state, action, reward, next_state):
def update_q_func(q_weights, state, action, reward, next_state):
    gamma = 0.99
    alpha = 0.5
    next_Max_Q=max(q_table[next_state])
    q_table[state, action] = (1 - alpha) * q_table[state, action] +\
            alpha * (reward + gamma * next_Max_Q)

    return q_table

# [4] start main function. set parameters
def q_learning(sensor_array):
    #[0] initialize
    action_num = 1
    ir_num = 1
    facial_num = 5
    other_teensy_num = 3

    state_number = ir_num + facial_num + other_teensy_num

    q_function_dim = 5
    num_episodes = 1000  #number of all trials

    action_array=np.zeros(action_num)
    q_weight = np.random.rand((q_function_dim, state_number))

    total_reward_vec = np.zeros(num_consecutive_iterations)  #contains rewards of each trial
    islearned = 0  #flg of finishing learning
    isrender = 0  #flg of drawing

    # [5] main tourine
    for episode in range(num_episodes):  #repeat for number of trials
        #initialize enviroment
        observation = get_sensor(sensor_array)
        #TODO
        action = q_max(state, q_weight)
        #ここでアクションする

        facial_expression = get_facial()
        reward = calc_reward(facial_expression)

        observation = sensor_input_dummy(sensor_array)
        next_state = digitize_state(observation)  #convert s_{t+1} to digitized value
        q_table = update_Qtable(q_table, state, action, reward, next_state)

    if islearned:
        np.savetxt('final_x.csv', final_x, delimiter=",")

if __name__ == '__main__':
    sensor_array=np.array([1,0.5,0.2])
    q_learning(sensor_array)
