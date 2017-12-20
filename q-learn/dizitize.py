# coding:utf-8
# http://neuro-educator.com/rl1/
# you need sudo pip install gym

# [0]import libraly
import numpy as np
import time

num_face = 20 #5%
num_ir = 20 #5mm
num_action = 6 #10deg

# [1]define Q-function
# convert observed situation to discrete values
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

# c nvert描画フラグ each values to discrete values
def digitize_state(state):
    neutral, happy, surprise, angry, sad, ir = state
    digitized = [
        np.digitize(neutral, bins=bins(0, num_face*5, num_face)),
        np.digitize(happy, bins=bins(0, num_face*5, num_face)),
        np.digitize(surprise, bins=bins(0, num_face*5, num_face)),
        np.digitize(angry, bins=bins(0, num_face*5, num_face)),
        np.digitize(sad, bins=bins(0, num_face*5, num_face)),
        np.digitize(ir, bins=bins(0,num_ir*5, num_ir)),
    ]
    return digitized

def digitize_action(action):
    d_action = np.digitize(action,bins=bins(0,num_action,num_action))
    return d_action
    #return sum([x * (num_dizitized**i) for i, x in enumerate(digitized)])

#num_dizitized = 4

#observation = np.array([1,2,0.5,2])
#print(digitize_state(observation))
