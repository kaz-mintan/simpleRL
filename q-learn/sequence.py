# coding:utf-8

import numpy as np

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
        reward = np.mean(np.dot(h,face),axis=1))

    elif mode == 'predict':
        e_face = face_predict - face
        reward = np.mean(e_face)

    return reward

def seq2feature(state):
    state_feature = np.mean(state, axis=1)
    return state_feature


