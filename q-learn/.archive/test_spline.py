from scipy.interpolate import RegularGridInterpolator
import numpy as np


def f(x, y, z):
    return 2 * x**3 + 3 * y**2 - z

def f2(x, y):
    return 2 * x**3 + 3 * y**2

x = np.linspace(1, 4, 2)
y = np.linspace(4, 7, 2)
z = np.linspace(7, 9, 2)

data2 = f2(*np.meshgrid(x, y, indexing='ij', sparse=True))
data = f(*np.meshgrid(x, y, z, indexing='ij', sparse=True))

#print(np.meshgrid(x, y, indexing='xy', sparse=True))
print('data',data2)

my_interpolating_function = RegularGridInterpolator((x, y, z), data)
my_interpolating_function2 = RegularGridInterpolator((x, y), data2)
print('x,y',(x,y))

pts = np.array([[1, 4, 7], [4, 7, 9]])
pts2 = np.array([[1, 4], [4, 7]])
print(my_interpolating_function2(pts2))
