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
    selected_num = present_face.argmax()
    #[1] choice the type of increasing facial expression
    #selected_num = random.randint(0,4)
    print(selected_num)
    dt_face=np.random.uniform(low=-5,high=5,size=1)
    face_pt_array[selected_num]+=dt_face

    print('selected, dt_face',selected_num,dt_face)
    print('first_array',face_pt_array)

    if face_pt_array[selected_num]-face_pt_array[selected_num-1] >= limit_max:
        #face_pt_array[selected_num]-=dt_face
        face_pt_array[selected_num]=limit_max
    elif face_pt_array[selected_num]-face_pt_array[selected_num-1] <= limit_min:
        #face_pt_array[selected_num]+=dt_face
        face_pt_array[selected_num]+=dt_face

    #[2] devide the remaining value of facial expression
    max_val = abs(dt_face)
    dev_num=np.zeros(array_size-1)
    dev_array=np.zeros(array_size-1)
    for i in range(array_size-1):
        dev_num[i]=random.uniform(0,max_val)
    dev_sort=np.sort(dev_num)

    sign = dt_face/max_val

    dev_array[0]=sign*(dev_sort[1]-dev_sort[0])

    for j in range(1,array_size-1):
        dev_array[j]=(dt_face/max_val)*(dev_sort[j]-dev_sort[j-1])


    count = 0
    for face_num in range(0,array_size):
        if face_num!=selected_num:
            face_pt_array[face_num]-=dev_array[count]
            print('dev_array[',count,']',dev_array[count],'face_pt_array[',face_num,']',face_pt_array[face_num])
            count=count+1

    print('face_pt_array',face_pt_array)

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
    loop_val = 2 
    face_val = np.array([100,0,0,0,0])
    face_max = 100
    face_min = 0
    facial=np.zeros((loop_val,5))
    for i in range(loop_val):
        facial[i,:]=face_val
        print(face_val)
        face_val= assumed_face(face_val,face_max,face_min)

    np.savetxt('test_face.csv',face_val,delimiter=',')
