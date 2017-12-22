# coding:utf-8

import numpy as np
import random

num_face = 5

def type2num(str_type):
    if str_type == 'neutral':
        num = 0
    elif str_type == 'happy':
        num = 1
    elif str_type == 'surprised':
        num = 2
    elif str_type == 'angry':
        num = 3
    elif str_type == 'sad':
        num = 4
    else:
        print('un-registered str')
        num = None
    return num

def get_face(action, target_type, nega_posi):
    pwm, time = action#2nd ver
    theta = pwm
    #print('dummy_evaluator.py/action',action)
    #print('dummy_evaluator.py/theta',theta)
    dummy_face = np.zeros(num_face)
    dev = np.zeros(num_face)
    dev[0]=0

    num = type2num(target_type)
    if nega_posi == 'posi':
        dummy_face[num]=2.5 * theta
    elif nega_posi == 'nega':
        dummy_face[num]=100-2.5 * theta
    if int(100.0 - dummy_face[num]) > num_face-1 :
        dev[1:]=random.sample(xrange(int(100.0-dummy_face[num])),num_face-1)
        dev_sort = np.sort(dev)

        t = 0
        #print('dev',dev_sort)
        for i in range(num_face):
            if i!=num:
                dummy_face[i]=dev_sort[t+1]-dev_sort[t]
                t+=1

    return dummy_face

if __name__ == '__main__':

    for theta in range(40):
        face=get_face(theta,'happy')
        print(face)

