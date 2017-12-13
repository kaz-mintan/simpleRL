# -*- coding: utf-8 -*- 
import numpy as np
import itertools

def make_pow_array(vector, weight):
    array = np.zeros((vector.size,weight.shape[0]))

    for i, j in itertools.product(range(vector.size), range(weight.shape[0])):
        new_num = pow(vector[i],j)
        array[i,j] = new_num
    return array

def calc_q_reward(observation, weight):
    array=make_pow_array(observation,weight)
    reward = 0

    for i in range(array.shape[0]):
        reward += np.dot(array[i,:],weight[i,:])
    return reward
    #print(reward)

def dif_q_func(state, action, weight):
    #return differential value

def q_function(state, action, weight):
    observation = np.hstack((state, action))
    if observation.size != weight.shape[0]:
        print('observation.size and weight.shape[0] are different!')
        return None
    else:
        reward = calc_q_reward(observation, weight)
    return reward

if __name__ == "__main__" :
    observation = np.array([2,3])
    action = np.array([2])
    weight = np.array([[1,2,3],[2,3,4],[0,0,0]])
    reward = q_function(observation,action, weight)
    print('reward',reward)

    #calc_q_reward(observation, weight)
    #make_pow_array(observation)

