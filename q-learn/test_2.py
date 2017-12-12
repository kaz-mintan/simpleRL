# coding:utf-8
# http://neuro-educator.com/rl1/

import numpy as np
import time
from dummy import *

# [1]define Q-function
def q_function(state, action, weights):
    ir_sensor, facial, pwm = state
    ir_weight, facial_weight, pwm_weight = weights
    q_value=np.dot(ir_sensor,ir_weight)
        +np.dot(facial,facial_weight)
        +np.dot(pwm,pwm_weight)

    return q_value

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
def update_Qtable(q_table, state, action, reward, next_state):
    gamma = 0.99
    alpha = 0.5
    next_Max_Q=max(q_table[next_state])
    q_table[state, action] = (1 - alpha) * q_table[state, action] +\
            alpha * (reward + gamma * next_Max_Q)

    return q_table

def q_learning(sensor_array):
    # [4] start main function. set parameters
    action_array=np.array([0.1,0.2,0.3])
    max_number_of_steps = 100  #number of steps for 1 trial
    num_consecutive_iterations = 100  #mean of number of trial to use for evaluation of finish of learning
    num_episodes = 1000  #number of all trials
    goal_average_reward = 50  #boder line of rewards to stop learning
    q_table = np.random.uniform(low=-1, high=1, size=4)
    #４じすぷらいんほかんてきな

    total_reward_vec = np.zeros(num_consecutive_iterations)  #contains rewards of each trial
    final_x = np.zeros((num_episodes, 1))  #contains a value of x (t=200) after learning
    islearned = 0  #flg of finishing learning
    isrender = 0  #flg of drawing

    # [5] main tourine
    #sensor_array=np.array([1,0.5,0.2])
    list_weight=np.array([1,2,3])
    for episode in range(num_episodes):  #repeat for number of trials
        #initialize enviroment
        observation = sensor_input_dummy(sensor_array)
        state = digitize_state(observation)
        #TODO
        action = np.argmax(q_table[state])
        #ここでアクションする

        facial_expression = get_facial()
        reward = calc_reward_dummy(facial_expression)
        done=done_dummy()

        observation = sensor_input_dummy(sensor_array)
        next_state = digitize_state(observation)  #convert s_{t+1} to digitized value
        q_table = update_Qtable(q_table, state, action, reward, next_state)

    if islearned:
        np.savetxt('final_x.csv', final_x, delimiter=",")

if __name__ == '__main__':
    sensor_array=np.array([1,0.5,0.2])
    q_learning(sensor_array)
