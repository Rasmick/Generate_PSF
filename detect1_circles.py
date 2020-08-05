
import numpy as np
from scipy.optimize import curve_fit
import scipy
import matplotlib.pyplot as plt
from fitting_2D_gaussian import gaussian
from fitting_2D_gaussian import moments
from fitting_2D_gaussian import fitgaussian
#from Fit_2D_gaussian import getFWHM_GaussianFitScaledAmp
#from Fit_2D_gaussian import twoD_GaussianScaledAmp
from numpy import pi, sqrt, exp
#import napari



from skimage import data, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.io import imread

def Gauss_1D(sigma, x,mu):
    gauss_fit=exp(-0.5*((x-mu)/sigma)^2)/(sigma*sqrt(2*pi))


# Load picture and detect edges
# image = img_as_ubyte(data.coins()[0:95, 70:370])
#image = imread('MMStack_Default.ome.tif')#[..., 0]
#image = imread('bead_image_tri2_0000.tif')

image = imread('MMStack_default_try.tif')
image_xy=image[1:100,:,500]
import napari
with napari.gui_qt():
    viewer = napari.view_image(image_xy, name='beads')
    viewer.theme = 'light'
    #layer = viewer.add_image(image_xy)

x_all=viewer.layers
x=viewer.layers[1]

rows, cols = 4,2
arr = [[0 for i in range(cols)] for j in range(rows)]
#print(type(arr))
arr=(np.round(x.data[0]))
#arr0=np.round(x.data[0][0])
#print(type(arr))
print(arr)
#print(arr[2][1])
x1=int(arr[0][0])
x2=int(arr[1][0])
y1=int(arr[0][1])
y2=int(arr[2][1])
print(x1,x2,y1,y2)

#print(type(arr))
#image_xyy=image_xy[13:59,800:1000]
#row, col=image_xyy.shape
#print(row,col)
#image_xy_crop=image_xy[22:60,100:500]
#image_xy_crop=image_xy[int(arr[0][0]):int(arr[1][0]),int(arr[0][0]):int(arr[2][1])]
image_xy_crop=image_xy[x1:x2,y1:y2]
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(20, 10))
ax.imshow(image_xy_crop, cmap=plt.cm.gray)
plt.show()

edges = canny(image_xy_crop, sigma=3, low_threshold=10, high_threshold=50)



# Detect two radii
hough_radii = np.arange(3, 5, 2)
hough_res = hough_circle(edges, hough_radii)

centers = []
accums = []
radii = []
height=[]
width_x=[0]*10
width_y=[0]*10

for radius, h in zip(hough_radii, hough_res):
    # For each radius, extract two circles
    num_peaks = 3
    peaks = peak_local_max(h, num_peaks=num_peaks)
    centers.extend(peaks)
    accums.extend(h[peaks[:, 0], peaks[:, 1]])
    radii.extend([radius] * num_peaks)

# Draw the most prominent 5 circles
image_color = color.gray2rgb(image_xy_crop)
for idx in np.argsort(accums)[::-1][:]:
    center_x, center_y = centers[idx]
    radius = radii[idx]
    cx, cy = circle_perimeter(center_y, center_x, radius)
    image_color[cy, cx] = (220, 20, 20)
    bead_image = image_xy_crop[min(cy) - 5:max(cy) + 5, min(cx) - 10:max(cx) + 5]
    x_len, y_len=np.shape(bead_image)
    #result = np.where(bead_image == np.amax(np.amax(bead_image)))
    #index=np.index(np.amax(np.amax(bead_image)))
    #print(bead_image(7,13))
    #scipy.optimize.curve_fit(Gauss_1D,np.arange(x_len),bead_image[x_index,:])
    bead_image_color=image_color[min(cy)-5:max(cy)+5,min(cx)-5:max(cx)+5]
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 10))
    ax.imshow(bead_image_color, cmap=plt.cm.gray)
    #plt.show()
    params = fitgaussian(bead_image)
    #FWHM_X, FWHM_Y = getFWHM_GaussianFitScaledAmp(bead_image)
    fit = gaussian(*params)
    (height, x, y, width_x[idx], width_y[idx]) = params
    print(width_x[idx],width_y[idx])
    #plt.contour(fit(*np.indices(bead_image.shape)), cmap=plt.cm.copper)
    #ax = plt.gca()
    #plt.show()
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(20, 10))
ax.imshow(image_xy_crop, cmap=plt.cm.gray)
plt.show()





