# coding:utf-8

import numpy as np
import random

def prob_output(output_a,output_b):
    num = random.randint(0,100)
    if num%2 == 0:
        return output_a
    else:
        return output_b

def evaluator():
    
    return eval_face

def assumed_face(present_face, limit_max, limit_min):
    array_size = present_face.size
    face_pt_array = np.array([
            present_face[0],
            present_face[0]+present_face[1],
            present_face[0]+present_face[1]+present_face[2],
            present_face[0]+present_face[1]+present_face[2]+present_face[3],
            present_face[0]+present_face[1]+present_face[2]+present_face[3]+present_face[4]]
            )

    dt_face=np.random.uniform(low=-1,high=1,size=1)
    face_pt_array[0]+=dt_face
    if face_pt_array[0] >= limit_max:
        face_pt_array[0]-=dt_face
    elif face_pt_array[0] <= limit_min:
        face_pt_array[0]+=dt_face
    print('face_dt',face_pt_array[0])

    for face_num in range(1,array_size):
        dt_face=np.random.uniform(low=-2,high=2,size=1)
        face_pt_array[face_num]+=dt_face
        print('face_dt',face_pt_array[face_num])
        #if assumed_z >= limit_max:
        if face_pt_array[face_num] >= limit_max-face_pt_array[face_num-1]:
        #if face_pt_array[face_num] >= limit_max:
            prob_output(face_pt_array[face_num]-dt_face,limit_max-face_pt_array[face_num-1])
            #face_pt_array[face_num]-=dt_face
            #face_pt_array[face_num]=limit_max-face_pt_array[]
        elif face_pt_array[face_num] <= limit_min+face_pt_array[face_num-1]:
            #face_pt_array[face_num]+=dt_face
            prob_output(face_pt_array[face_num]+dt_face,limit_max+face_pt_array[face_num-1])

    assumed_face = np.array([
            face_pt_array[0],
            face_pt_array[1]-face_pt_array[0],
            face_pt_array[2]-face_pt_array[1],#-face_pt_array[0],
            face_pt_array[3]-face_pt_array[2],#-face_pt_array[1]-face_pt_array[0],
            face_pt_array[4]-face_pt_array[3]]#-face_pt_array[2]-face_pt_array[1]-face_pt_array[0]]
     )
    print('last face array', assumed_face)

    return assumed_face

if __name__ == '__main__':
    loop_val = 4 
    face_val = np.array([100,0,0,0,0])
    face_max = 100
    face_min = 0
    facial=np.zeros((loop_val,5))
    for i in range(loop_val):
        facial[i,:]=face_val
        print(face_val)
        face_val= assumed_face(face_val,face_max,face_min)

    np.savetxt('test_face.csv',face_val,delimiter=',')
