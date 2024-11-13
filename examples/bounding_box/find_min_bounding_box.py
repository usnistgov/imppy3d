"""
Written by Newell Moser
NRC Postdoctoral Fellow
National Institute of Standards and Technology
February 28, 2023

Generate some example 3D data points (in Cartesian space), and calculate
the minimum fitting bounding box based on minimizing the volume. The
results are plotted as well.
"""

# Import external dependencies
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

import imppy3d.import_export as imex
import imppy3d.cv_processing_wrappers as wrap
import imppy3d.cv_driver_functions as drv
import imppy3d.bounding_box as box

# Set constants related to plotting (for MatPlotLib)
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=MEDIUM_SIZE)         # Controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # Fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # Fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # Fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # Fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # Legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # Fontsize of the figure title


def connect_box_corners(corner_pnts):
    """
    Helper function that connects all of the corners of a 3D bounding
    box into a single, continuous line for easier plotting. The order
    of the corners is based on the BBox class defined in
    volume_image_processing.py
    """
    return np.array([
        corner_pnts[0,:],
        corner_pnts[1,:],
        corner_pnts[2,:],
        corner_pnts[3,:],
        corner_pnts[0,:],
        corner_pnts[4,:],
        corner_pnts[5,:],
        corner_pnts[1,:],
        corner_pnts[5,:],
        corner_pnts[6,:],
        corner_pnts[2,:],
        corner_pnts[6,:],
        corner_pnts[7,:],
        corner_pnts[3,:],
        corner_pnts[7,:],
        corner_pnts[4,:] ])


# -------- MAKE SOME EXAMPLE DATA TO TEST --------

# data1 ==> Two crosses on top of each other in the XY-planes
data1 = np.array([
    [5, -10, -2],
    [5, -10, 1],
    [5, -2, -2],
    [5, -2, 1],
    [0, -7, -2],
    [0, -7, 1],
    [9, -7, -2],
    [9, -7, 1] ])

# data2 ==> Random data in a three dimensions
n_rand_pnts = 500

n_half = (np.floor(n_rand_pnts/2)).astype(np.int32)
data2 = (np.random.random_sample((n_rand_pnts,3)))
data2[0:n_half,0] = data2[0:n_half,0]*8
data2[0:n_half,1] = data2[0:n_half,1]*4
data2[0:n_half,2] = data2[0:n_half,2]*2
data2[n_half:n_rand_pnts,0] = data2[n_half:n_rand_pnts,0]*2 + 2.0
data2[n_half:n_rand_pnts,1] = data2[n_half:n_rand_pnts,1]*4 + 2.0
data2[n_half:n_rand_pnts,2] = data2[n_half:n_rand_pnts,2]*8 + 2.0

# data3 ==> Tetrahedron (of edge length 2) 
data3 = np.array([
    [1.0, 0.0, -1.0/np.sqrt(2)],
    [-1.0, 0.0, -1.0/np.sqrt(2)],
    [0.0, 1.0, 1.0/np.sqrt(2)],
    [0.0, -1.0, 1.0/np.sqrt(2)] ])

# data4 ==> Ellipsoid (via spherical parameterization)
ang_step = np.pi/18.0
theta = np.arange(0.0, np.pi, ang_step)
psi = np.arange(0.0, 2.0*np.pi, ang_step)
ax_a = 12.0
ax_b = 6.0
ax_c = 3.0

num_pnts = theta.shape[0] * psi.shape[0]
data4 = np.zeros((num_pnts, 3))
cur_i = 0
for cur_th in theta:
    for cur_ps in psi:
        data4[cur_i,0] = ax_a*np.sin(cur_th)*np.cos(cur_ps)
        data4[cur_i,1] = ax_b*np.sin(cur_th)*np.sin(cur_ps)
        data4[cur_i,2] = ax_c*np.cos(cur_th)
        cur_i += 1

# data5 ==> Double sphere (different diameters) with a little overlap
# Did not finish the double-sphere data set


# ---------- SELECT THE DATA SET TO TEST ---------- 

# To change to a different data set, set test_data equal either to:
# data1: Two crosses on top of each other in the XY-planes
# data2: Random data in a three dimensions
# data3: Tetrahedron of edge length 2
# data4: Ellipsoid
test_data = data2


# ---------- APPLY A ROTATION AND TRANSLATION THE DATA ----------

# Set to false if you do not want to perform random rotation
# and translation to the data set
calc_rand_transform = True

if calc_rand_transform:
    x_tran = np.random.rand(1)*10.0
    y_tran = np.random.rand(1)*10.0
    z_tran = np.random.rand(1)*10.0
    x_deg = np.random.rand(1)*180.0
    y_deg = np.random.rand(1)*180.0
    z_deg = np.random.rand(1)*180.0

    x_rad = np.deg2rad(x_deg[0])
    y_rad = np.deg2rad(y_deg[0])
    z_rad = np.deg2rad(z_deg[0])

    cx = np.cos(x_rad)
    sx = np.sin(x_rad)
    R_x = np.array([ 
        [1., 0., 0.],
        [0., cx, -sx],
        [0., sx, cx] ])

    cy = np.cos(y_rad)
    sy = np.sin(y_rad)
    R_y = np.array([
        [cy, 0., sy],
        [0., 1., 0.],
        [-sy, 0., cy] ])

    cz = np.cos(z_rad)
    sz = np.sin(z_rad)
    R_z = np.array([
        [cz, -sz, 0.],
        [sz, cz, 0.],
        [0., 0., 1.] ])

    R_zyx = np.matmul(R_z, np.matmul(R_y, R_x))

    # Finally, apply random 3D rotation
    test_data = np.dot(R_zyx, np.transpose(test_data))
    test_data = np.transpose(test_data)

    # Apply random translation
    test_data = test_data + np.array([x_tran[0], x_tran[0], x_tran[0]])


# -------- CALCULATE MINIMUM VOLUME BOUNDING BOX --------

print("\nCalculating bounding box...")

# Set "search" to either 0, 1, 2, 3, 4 to trial different bounding
# box algorithms
bbox_obj = box.min_bounding_box(test_data, search=4)
print("  Done!")


# -------- OUTPUT TO THE TERMINAL SOME USEFUL LOG INFORMATION --------

print(f"\nMean point of data set: {np.mean(test_data, axis=0)}")

bbox_corners = bbox_obj.corners
bbox_lines = connect_box_corners(bbox_corners)
print("\nBounding box properties:")
print(f"  Local X-length: {bbox_obj.x_len}")
print(f"  Local Y-length: {bbox_obj.y_len}")
print(f"  Local Z-length: {bbox_obj.z_len}")
print(f"  Volume: {bbox_obj.volume}")
print(f"  Center point: {bbox_obj.center}")
print(f"\n  Rotation matrix: \n{bbox_obj.R_zyx}\n")
print(f"  Corner coordinates: \n{bbox_obj.corners}\n")

# Can check if the bounding box is regular (90 deg corners)
# by calculating the diagonals -- they should all be equal
diag1 = (bbox_corners[0,0]-bbox_corners[6,0])**2 + \
        (bbox_corners[0,1]-bbox_corners[6,1])**2 + \
        (bbox_corners[0,2]-bbox_corners[6,2])**2
diag1 = np.sqrt(diag1)

diag2 = (bbox_corners[1,0]-bbox_corners[7,0])**2 + \
        (bbox_corners[1,1]-bbox_corners[7,1])**2 + \
        (bbox_corners[1,2]-bbox_corners[7,2])**2
diag2 = np.sqrt(diag2)

diag3 = (bbox_corners[2,0]-bbox_corners[4,0])**2 + \
        (bbox_corners[2,1]-bbox_corners[4,1])**2 + \
        (bbox_corners[2,2]-bbox_corners[4,2])**2
diag3 = np.sqrt(diag3)

diag4 = (bbox_corners[3,0]-bbox_corners[5,0])**2 + \
        (bbox_corners[3,1]-bbox_corners[5,1])**2 + \
        (bbox_corners[3,2]-bbox_corners[5,2])**2
diag4 = np.sqrt(diag4)

print("  Diagonal lengths (which should all be equal) are: ")
print(f"    Diagonals 1=2=3=4: {diag1:.6f} = {diag2:.6f} = " + \
    f"{diag3:.6f} = {diag4:.6f}")


# -------- SHOW 3D PLOTS --------

# Need to do some additional calculations to ensure equal axes
x_range = np.amax(bbox_corners[:,0]) - np.amin(bbox_corners[:,0])
y_range = np.amax(bbox_corners[:,1]) - np.amin(bbox_corners[:,1])
z_range = np.amax(bbox_corners[:,2]) - np.amin(bbox_corners[:,2])
xyz_range = np.array([x_range, y_range, z_range])
i_max = np.argmax(xyz_range)
lim_half = xyz_range[i_max]/2.0

cent_xyz = bbox_obj.center

print("\nMaking plots...")
print("\nWARNING: 3D plots using matplotlib cannot easily be drawn with " +\
    "equal axis lengths.\nThe bounding box may appear to have skewed " +\
    "angles, but in fact, does have 90 deg corners.")
print("\nClose the figure to end this script...")
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(test_data[:,0], test_data[:,1], test_data[:,2], \
    marker='o', color='black')
ax.plot(bbox_lines[:,0], bbox_lines[:,1], bbox_lines[:,2],\
    linestyle='--')

ax.set_xlim(cent_xyz[0] - lim_half, cent_xyz[0] + lim_half)
ax.set_ylim(cent_xyz[1] - lim_half, cent_xyz[1] + lim_half)
ax.set_zlim(cent_xyz[2] - lim_half, cent_xyz[2] + lim_half)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

print("\nScript finished successfully!")