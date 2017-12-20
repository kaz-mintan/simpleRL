# coding:utf-8

import numpy as np
import sys


num_face = 5
num_ir = 1

# reward function
def calc_reward(state, state_predict, time_window, mode):
    # coefficient
    c = np.array([1,1,1,-1,-1]) #for delta mode
    h = np.array([1,1,1,-1,-1]) #for heuristic mode
    reward = 0

    # extract face array (must be time sequence data)
    face = state[0:num_face,:] #in numpy, the 5 of the 0:5 is not included
    face_post = face[:,1:] #for delta mode
    face_predict = state_predict[0:num_face,:] #for predict mode

    #return sum([x * (num_dizitized**i) for i, x in enumerate(digitized)])
    if mode == 'delta':
        c_face=np.zeros((num_face,time_window-1))
        d_face = face_post - face[:,:time_window-2]
        for face_type in range(num_face):
            c_face[face_type,:]=c[face_type]*d_face[face_type,:]
        reward = np.mean(c_face)

    elif mode == 'heuristic':
        h_face=np.zeros((num_face,time_window))
        for face_type in range(num_face):
            h_face[face_type,:]=h[face_type]*face[face_type,:]
        reward = np.mean(h_face)

    elif mode == 'predict':
        e_face = face_predict - face
        reward = np.mean(e_face)

    return reward

def seq2feature(state):
    state_feature = np.mean(state, axis=1)
    return state_feature

if __name__ == "__main__" :
    argvs = sys.argv  # コマンドライン引数を格納したリストの取得
    time_window = 3
    mode = argvs[1]
    state = np.random.uniform(low=0,high=1,size=(num_face+num_ir,time_window))
    state_predict = np.random.uniform(low=0,high=1,size=(num_face+num_ir,time_window))
    #print('state_predict',state_predict)

    reward = calc_reward(state, state_predict, time_window, mode)
    print('reward',reward)

    feature_state = seq2feature(state)
    print(feature_state)
