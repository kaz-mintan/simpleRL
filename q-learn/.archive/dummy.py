m coding:utf-8
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

def actuator_test(motor_array):
    present_angl, direction, velosity = motor_array
    #for i in range():
    present_angl+=direction*velosity
    return present_angl

class Q_func_dummy():
    def __init__(self):
        self.name=""

    def qvalue(self, list_weight, state, action):
        self.q_value=randint(50)
        return self.q_value

    def qmax_action(self, list_weight, state, action):
        self.qmax=randint(50)
        self.max_action=np.random.uniform(low=0,high=1,size=action.size)
        return self.max_action

def digitize_action(action):
    digitized = np.digitize(action, bins=bins(0, 1, num_dizitized)),
    return digitized

if __name__ == '__main__':
    sensor_array=np.array([1,0.5,0.2])
    facial_array=np.array([1,0.5,0.2,0.2,0.1])
    print("sensor",sensor_input_dummy(sensor_array))
    print("facial",facial_input_dummy())
    print("reward",calc_reward_dummy(facial_input_dummy()))
    print(done_dummy())
    motor_array=np.array([100,1,0.2])
    print("motor",actuator_test(motor_array))

    list_weight=np.array([1,2,3])
    action=np.array([0.1,0.2,0.3])

    q_func=Q_func_dummy()
    value=q_func.qvalue(list_weight,sensor_array,action)
    print("q_value", value)
    print("qmax_action", q_func.qmax_action(list_weight,sensor_array,action))
