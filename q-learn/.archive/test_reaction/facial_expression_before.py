# coding:utf-8

import numpy as np
import random

def prob_output(output_a,output_b):
    num = random.randint(0,100)
    if num%2 == 0:
        return output_a
    else:
        return output_b

def assumed_face(present_face, limit_max, limit_min):
    array_size = present_face.size
    face_pt_array = present_face

    selected_num = present_face.argmax()
    #[1] choice the type of increasing facial expression
    before_face = face_pt_array[selected_num]
    dt_face=np.random.randint(-5,5,1)
    face_pt_array[selected_num]+=dt_face

    if face_pt_array[selected_num] >= limit_max:
        face_pt_array[selected_num]=prob_output(limit_max,face_pt_array[selected_num]-dt_face)
    elif face_pt_array[selected_num]-face_pt_array[selected_num-1] <= limit_min:
        face_pt_array[selected_num]=prob_output(limit_min,face_pt_array[selected_num]-dt_face)

    dt_face = face_pt_array[selected_num]-before_face

    #[2] devide the remaining value of facial expression
    dev_num=np.zeros(array_size)
    dev_array=np.zeros(array_size-1)

    if dt_face!=0:
        max_val = abs(dt_face)
        print('dt_face',dt_face)
        dev_num[array_size-1]=max_val

        if max_val < array_size:
            dev_num[1:max_val]=random.sample(xrange(max_val), max_val-1)
        else:
            dev_num[1:array_size-1]=random.sample(xrange(max_val), array_size-2)

        dev_sort=np.sort(dev_num)
        #print('dev_sort',dev_sort)

        sign = dt_face/max_val

        dev_array[0]=sign*(dev_sort[1]-dev_sort[0])

        for j in range(1,array_size-1):
            dev_array[j]=(dt_face/max_val)*(dev_sort[j+1]-dev_sort[j])


    count = 0
    for face_num in range(0,array_size):
        if face_num!=selected_num:
            face_pt_array[face_num]-=dev_array[count]
            if face_pt_array[face_num]<limit_min:
                diff = limit_min - face_pt_array[face_num]
                face_pt_array[face_num]=limit_min

                for buf in range(0,face_num):
                    if buf!=selected_num:
                        if face_pt_array[buf]>diff:
                            print('buf', buf,'diff',diff)
                            face_pt_array[buf]-=diff
                            diff=0
                        else:
                            face_pt_array[buf]=limit_min
                    else:
                        print('if')

            if face_pt_array[face_num]>limit_max:
                print('up')

            count=count+1

    return face_pt_array

if __name__ == '__main__':
    loop_val = 20
    face_val = np.array([100,0,0,0,0])
    face_max = 100
    face_min = 0
    facial=np.zeros((loop_val,5))
    for i in range(loop_val):
        facial[i,:]=face_val
        print(face_val, face_val.sum())
        face_val= assumed_face(face_val,face_max,face_min)

    np.savetxt('test_face.csv',face_val,delimiter=',')
