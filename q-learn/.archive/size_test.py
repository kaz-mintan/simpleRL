# coding:utf-8
import numpy as np

num_dizitized = 6  #number of digitized/devided

q_table = np.random.uniform(
    low=0, high=1, size=(3, 2))
    #low=-1, high=1, size=(num_dizitized**4, 2))

def get_face():
    return np.array([2,2,1,2,1])

def get_ir():
    return np.array([1])

face = get_face()
ir = get_ir()
state = np.hstack((get_face(), get_ir()))
print(state)

print(q_table)

