# coding:utf-8
import numpy as np
from numpy.random import *

def sensor_input_dummy(array):
    sensor_info=np.random.uniform(low=0,high=1,size=array.size)
    return sensor_info


def facial_input_dummy():
    num=5
    facial_info=np.random.uniform(low=0,high=1,size=num)
    return facial_info

def calc_reward_dummy(facial):
    weight_array=np.array([0,100,40,-50,-50])
    #[neutral,happy,surprise,angry,sad]
    reward=np.dot(facial,weight_array)
    return reward

def done_dummy():
    done_bool = randint(100)%2
    return done_bool

if __name__ == '__main__':
    sensor_array=np.array([1,0.5,0.2])
    facial_array=np.array([1,0.5,0.2,0.2,0.1])
    print("sensor",sensor_input_dummy(sensor_array))
    print("facial",facial_input_dummy())
    print("reward",calc_reward_dummy(facial_input_dummy()))
    print(done_dummy())

