import numpy as np
from numpy import pi, r_
import matplotlib.pyplot as plt
from scipy import optimize


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(0.5*
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    #X=np.multiply(X,2)
    #Y=np.multiply(Y,148*10^-9)
    #print(type(X))
    #x_size, y_size =Y.shape
    #print((X[3][1]))
    #print(x_size, y_size)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    pixel_size_x = 148*10**-9
    pixel_size_z = 200*10**-9
    print(type(x))
    col = data[:, int(y)]
    y1 = y*pixel_size_z
    x1 = y*pixel_size_x
    width_x = np.sqrt(np.abs(((np.arange(col.size))*pixel_size_z-y1)**2*col).sum()/col.sum())
    print(width_x)
    row = data[int(x), :]
    width_y = np.sqrt(np.abs(((np.arange(row.size))*pixel_size_x-x1)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p