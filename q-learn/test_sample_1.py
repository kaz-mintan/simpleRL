# coding:utf-8
# http://neuro-educator.com/rl1/

# [0]import libraly
#import gym  #cartpole
#from gym import wrappers  #save a pic of gym
import numpy as np
import time
from dummy import *

# [1]define Q-function
# convert observed situation to discrete values
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

# c nvert描画フラグ each values to discrete values
def digitize_state(observation):
    cart_pos, cart_v, pole_angle = observation
    #cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, num_dizitized)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, num_dizitized)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, num_dizitized)),
        #np.digitize(pole_v, bins=bins(-2.0, 2.0, num_dizitized))
    ]
    return sum([x * (num_dizitized**i) for i, x in enumerate(digitized)])

# [2]function which determine action a(t)
def get_action(next_state, episode, action):
    #epsilon-greedy
    epsilon = 0.5 * (1 / (episode + 1))
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        #next_action = np.random.choice([0, 1])
        next_action = np.randint(action.size)
    return next_action

# [3]function which update Q-table
def update_Qtable(q_table, state, action, reward, next_state):
    gamma = 0.99
    alpha = 0.5
    next_Max_Q=max(q_table[next_state])
    q_table[state, action] = (1 - alpha) * q_table[state, action] +\
            alpha * (reward + gamma * next_Max_Q)

    return q_table

# [4] start main function. set parameters
action_array=np.array([0.1,0.2,0.3])
max_number_of_steps = 100  #number of steps for 1 trial
num_consecutive_iterations = 100  #mean of number of trial to use for evaluation of finish of learning
num_episodes = 1000  #number of all trials
goal_average_reward = 50  #boder line of rewards to stop learning
# state is digitized/devided into 6 parts (there is 4 variables) and making q-table (as q-function)
num_dizitized = 6  #number of digitized/devided
q_table = np.random.uniform(    low=-1, high=1, size=(num_dizitized**4, action_array.size))

total_reward_vec = np.zeros(num_consecutive_iterations)  #contains rewards of each trial
final_x = np.zeros((num_episodes, 1))  #contains a value of x (t=200) after learning
islearned = 0  #flg of finishing learning
isrender = 0  #flg of drawing

# [5] main tourine
sensor_array=np.array([1,0.5,0.2])
list_weight=np.array([1,2,3])
for episode in range(num_episodes):  #repeat for number of trials
    # initialize enviroment
    observation = sensor_input_dummy(sensor_array)
    #observation = env.reset()
    state = digitize_state(observation)
    #TODO
    action = np.argmax(q_table[state])
    episode_reward = 0

    for t in range(max_number_of_steps):  #roup for 1 trial
        # calcurate s_{t+1}, r_{t} etc based on selected/conducted action
        observation = sensor_input_dummy(sensor_array)
        facial_expression = facial_input_dummy()
        reward = calc_reward_dummy(facial_expression)
        done=done_dummy()

        episode_reward += reward  #報酬を追加

        # calcurate s_{t+1} and update q-table(as q-function)
        next_state = digitize_state(observation)  #convert s_{t+1} to digitized value
        q_table = update_Qtable(q_table, state, action, reward, next_state)

        # evaluate the next action a_{t+1}
        action = get_action(next_state, episode, action)    # a_{t+1} 

        state = next_state

        # processing of end
        if done:
            print('%d Episode finished after %f time steps / mean %f' %
                  (episode, t + 1, total_reward_vec.mean()))
            total_reward_vec = np.hstack((total_reward_vec[1:],
                                          episode_reward))  #record a reward
            if islearned == 1:  #if learning has finished
                final_x[episode, 0] = observation[0] #contain the value of final x
            break

    if (total_reward_vec.mean() >=
            goal_average_reward):  # if the rewards of the most recent 100 episodes is larger than normal rewards, success 
        print('Episode %d train agent successfuly!' % episode)
        islearned = 1
        #np.savetxt('learned_Q_table.csv',q_table, delimiter=",") #if save Q-table
        if isrender == 0:
            #env = wrappers.Monitor(env, './movie/cartpole-experiment-1') #if save the movie
            isrender = 1

if islearned:
    np.savetxt('final_x.csv', final_x, delimiter=",")
