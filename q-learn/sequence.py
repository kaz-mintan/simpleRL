# coding:utf-8

import numpy as np
import sys

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
        reward = np.mean(np.dot(h,face),axis=1)

    elif mode == 'predict':
        e_face = face_predict - face
        reward = np.mean(e_face)

    return reward

def seq2feature(state):
    state_feature = np.mean(state, axis=1)
    return state_feature

if __name__ == "__main__" :
    argvs = sys.argv  # コマンドライン引数を格納したリストの取得

    time_window = 10
    num_face = 2
    num_ir = 2

    mode = argvs[1]

    state = np.random.uniform(low=0,high=1,size=(num_face*5+num_ir,time_window))
    print('state',state)
    state_predict = np.random.uniform(low=0,high=1,size=(num_face*5+num_ir,time_window))
    reward = calc_reward(state,state_predict, mode)
    print('reward',reward)

