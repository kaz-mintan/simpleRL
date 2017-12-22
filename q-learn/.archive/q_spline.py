# coding:utf-8
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def q_basefunc(s_points, a_points, q_points):
    my_interpolating_function = RegularGridInterpolator((s_points, a_points), q_points)
    return my_interpolating_function

if __name__ == "__main__" :
    state = np.array([0,5])
    action = np.array([10,7])
    q_value = np.array([4,2])


    func = q_basefunc(state,action,q_value)

    s =np.linspace(0,10,num=21)
    a =np.linspace(0,10,num=21)
    q = func(s,a)

    S,A,Q=np.meshgrid(s,a,q)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(S,A,Q) #<---ここでplot

    plt.show()
