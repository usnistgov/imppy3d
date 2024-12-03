"""
Written by Newell Moser
NRC Postdoctoral Fellow
National Institute of Standards and Technology
December 2, 2024

Tests for calcualating the minimum rotated bounding box of a point cloud.
"""

import os
import numpy as np
import imppy3d.bounding_box as box


from matplotlib import pyplot as plt
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


def farthest_dist_between_pnts(pnt_arr):
    num_pnt = pnt_arr.shape[0]
    n_dim = pnt_arr.shape[1]
    dist_arr = np.zeros((num_pnt*num_pnt, n_dim))

    dist_i = 0
    for m, pnt1 in enumerate(pnt_arr):
        for n in np.arange(start=m, stop=num_pnt):
            pnt2 = pnt_arr[n]
            dist_arr[dist_i] = np.sqrt(np.sum(np.power(pnt1 - pnt2, 2)))
            dist_i += 1

    return np.amax(dist_arr)


def check_if_pnt_in_arr(pnt_0, pnt_arr, TOL):
    num_pnt = pnt_arr.shape[0]
    n_dim = pnt_arr.shape[1]

    TOL2 = TOL*TOL

    for cur_pnt in pnt_arr:
        cur_dist = np.sum(np.power(pnt_0 - cur_pnt, 2))

        if cur_dist <= TOL2:
            return True

    return False


def create_dummy_pnts(L=16.0, W=8.0, T=4.0):
    # [X, Y, Z]
    pnt1 = np.array([0.0, 0.0, 0.0])
    pnt2 = np.array([L, 0.0, 0.0])
    pnt3 = np.array([L, W, 0.0])
    pnt4 = np.array([0.0, W, 0.0])
    pnt5 = np.array([0.0, 0.0, T])
    pnt6 = np.array([L, 0.0, T])
    pnt7 = np.array([L, W, T])
    pnt8 = np.array([0.0, W, T])
    #pnt9 = np.array([L/2, W/2, T/2])

    pnt_arr = np.vstack((pnt1, pnt2, pnt3, pnt4, pnt5, pnt6, pnt7, pnt8))

    z_deg = 45.0
    y_deg = 45.0
    x_deg = 45.0

    x_rad = np.deg2rad(x_deg)
    y_rad = np.deg2rad(y_deg)
    z_rad = np.deg2rad(z_deg)

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
    pnt_arr = np.dot(R_zyx, np.transpose(pnt_arr))
    pnt_arr = np.transpose(pnt_arr)

    return pnt_arr


def bbox_svd():
    test_name = "\nTEST: BOUNDING BOX FIT USING SVD SEARCH..."
    print(test_name)

    pnts_src = create_dummy_pnts()
    pnts_bool_arr = np.full(pnts_src.shape[0], False, dtype=bool)
    TOL = 0.05*farthest_dist_between_pnts(pnts_src)

    try:
        bbox_obj = box.min_bounding_box(pnts_src, calc_hull=False, search=3)
    except:
        print(f"\nERROR: Failed to fit a bounding box using SVD search.")
        return [1, test_name + " ERROR"]

    bbox_corners = bbox_obj.corners

    for m, cur_pnt_src in enumerate(pnts_src):
        temp_bool = check_if_pnt_in_arr(cur_pnt_src, bbox_corners, TOL)
        pnts_bool_arr[m] = temp_bool

    if not all(pnts_bool_arr):
        print(f"\nERROR: Bounding box not within tolerance of actual points.")
        return [1, test_name + " ERROR"]

    return [0, test_name + " SUCCESS"] # (False) No errors


def bbox_exhaustive():
    test_name = "\nTEST: BOUNDING BOX FIT USING EXHAUSTIVE SEARCH..."
    print(test_name)

    pnts_src = create_dummy_pnts()
    pnts_bool_arr = np.full(pnts_src.shape[0], False, dtype=bool)
    TOL = 0.05*farthest_dist_between_pnts(pnts_src)

    try:
        bbox_obj = box.min_bounding_box(pnts_src, calc_hull=False, search=0)
    except:
        print(f"\nERROR: Failed to fit a bounding box using exhaustive search.")
        return [1, test_name + " ERROR"]

    bbox_corners = bbox_obj.corners

    for m, cur_pnt_src in enumerate(pnts_src):
        temp_bool = check_if_pnt_in_arr(cur_pnt_src, bbox_corners, TOL)
        pnts_bool_arr[m] = temp_bool

    if not all(pnts_bool_arr):
        print(f"\nERROR: Bounding box not within tolerance of actual points.")
        return [1, test_name + " ERROR"]

    return [0, test_name + " SUCCESS"] # (False) No errors


def bbox_LWT():
    test_name = "\nTEST: BOUNDING BOX FIT USING LWT SEARCH..."
    print(test_name)

    pnts_src = create_dummy_pnts(L=512, W=8, T=4)
    pnts_bool_arr = np.full(pnts_src.shape[0], False, dtype=bool)
    TOL = 0.05*farthest_dist_between_pnts(pnts_src)

    try:
        bbox_obj = box.min_bounding_box(pnts_src, calc_hull=False, search=4)
    except:
        print(f"\nERROR: Failed to fit a bounding box using LWT search.")
        return [1, test_name + " ERROR"]

    bbox_corners = bbox_obj.corners

    print(f"\npnts_src = \n{pnts_src}")
    print(f"\nbbox_corners = \n{bbox_corners}")

    for m, cur_pnt_src in enumerate(pnts_src):
        temp_bool = check_if_pnt_in_arr(cur_pnt_src, bbox_corners, TOL)
        pnts_bool_arr[m] = temp_bool

    if not all(pnts_bool_arr):
        print(f"\nERROR: Bounding box not within tolerance of actual points.")
        return [1, test_name + " ERROR"]

    return [0, test_name + " SUCCESS"] # (False) No errors


if __name__ == '__main__':

    flag_300, msg_300 = bbox_svd()
    flag_301, msg_301 = bbox_exhaustive()
    flag_302, msg_302 = bbox_LWT()

    print(f"\n\n\n---------- SUMMARY ----------")
    print(msg_300)
    print(msg_301)
    print(msg_302)
    print("\n")