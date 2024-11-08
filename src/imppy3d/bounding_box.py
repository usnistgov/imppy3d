import numpy as np
from scipy import spatial as spt

from . import volume_image_processing as vol

class BBox:
    """
    This is a simple class that stores a number of attributes that helps
    to define a 3D box, which is used later on to represent a bounding
    box for a 3D data set. The box is presumed to be oriented in some
    direction in 3D space, but a 3D rotation can be performed to align
    edges of the box with a new, local XYZ-coordinate system. The 
    rotation matrix, R_zyx, can be applied to the corners of the box
    to rotate them into this local coordinate system via,

        np.dot(R_zyx, np.transpose(corners))

    Note, this rotation matrix is constructed through intrinsic Euler
    angles defined by x_rad, y_rad, and z_rad. The length, width, and
    thickness of the box is given by x_len, y_len, and z_len. However,
    these lengths are NOT sorted since they are based on the local
    coordinate system and not their individual magnitudes.
    """

    def __init__(self, x_rad=0., y_rad=0., z_rad=0.):
        self.x_rad = x_rad # Angle about X-axis (radians)
        self.y_rad = y_rad # Angle about Y-axis (radians)
        self.z_rad = z_rad # Angle about Z-axis (radians)
        self.x_len = 0.0 # Length along local X-direction
        self.y_len = 0.0 # Length along local Y-direction
        self.z_len = 0.0 # Length along local Z-direction
        self.volume = 0.0
        self.center = np.array([0.0, 0.0, 0.0])
        self.corners = np.zeros((8,3)) # Coordinates of all 8 corners
        self.R_zyx = np.zeros((3,3)) # Rotation matrix 

    def calc_Rzyx(self):
        # Build a rotation matrix based on z-y'-x'' intrinsic
        # Euler angles.

        cx = np.cos(self.x_rad)
        sx = np.sin(self.x_rad)
        R_x = np.array([ 
            [1., 0., 0.],
            [0., cx, -sx],
            [0., sx, cx] ])

        cy = np.cos(self.y_rad)
        sy = np.sin(self.y_rad)
        R_y = np.array([
            [cy, 0., sy],
            [0., 1., 0.],
            [-sy, 0., cy] ])

        cz = np.cos(self.z_rad)
        sz = np.sin(self.z_rad)
        R_z = np.array([
            [cz, -sz, 0.],
            [sz, cz, 0.],
            [0., 0., 1.] ])

        self.R_zyx = np.matmul(R_z, np.matmul(R_y, R_x))
        return self.R_zyx

    def calc_volume(self):
        self.volume = self.x_len*self.y_len*self.z_len
        return self.volume

    def calc_center(self):
        self.center = np.mean(self.corners, axis=0)
        return self.center

    def calc_x_vec(self):
        b_x_vec = self.corners[0,:] - self.corners[3,:]
        b_x_vec = b_x_vec / np.sqrt(np.power(b_x_vec,2).sum())
        return b_x_vec

    def calc_y_vec(self):
        b_y_vec = self.corners[2,:] - self.corners[3,:]
        b_y_vec = b_y_vec / np.sqrt(np.power(b_y_vec,2).sum())
        return b_y_vec

    def calc_z_vec(self):
        b_z_vec = self.corners[7,:] - self.corners[3,:]
        b_z_vec = b_z_vec / np.sqrt(np.power(b_z_vec,2).sum())
        return b_z_vec

    def copy_to_self(self, other):
        self.x_rad = other.x_rad
        self.y_rad = other.y_rad
        self.z_rad = other.z_rad
        self.x_len = other.x_len 
        self.y_len = other.y_len
        self.z_len = other.z_len 
        self.volume = other.volume
        self.center = other.center.copy()
        self.corners = other.corners.copy()
        self.R_zyx = other.R_zyx.copy() 


class BBox2D:
    """
    Same as BBox above, but a 2D simplification.
    """

    def __init__(self, z_rad=0.):
        self.z_rad = z_rad # Angle about Z-axis (radians)
        self.x_len = 0.0 # Length along local X-direction
        self.y_len = 0.0 # Length along local Y-direction
        self.area = 0.0
        self.center = np.array([0.0, 0.0])
        self.corners = np.zeros((4,2)) # Coordinates of all 4 corners
        self.R_z = np.zeros((2,2)) # Rotation matrix 

    def calc_Rz(self):
        # Build a 2D rotation matrix
        cz = np.cos(self.z_rad)
        sz = np.sin(self.z_rad)
        R_z = np.array([
            [cz, -sz],
            [sz, cz] ])

        self.R_z = R_z
        return self.R_z

    def calc_area(self):
        self.area = self.x_len*self.y_len
        return self.area

    def calc_center(self):
        self.center = np.mean(self.corners, axis=0)
        return self.center

    def calc_x_vec(self):
        b_x_vec = self.corners[0,:] - self.corners[3,:]
        b_x_vec = b_x_vec / np.sqrt(np.power(b_x_vec,2).sum())
        return b_x_vec

    def calc_y_vec(self):
        b_y_vec = self.corners[2,:] - self.corners[3,:]
        b_y_vec = b_y_vec / np.sqrt(np.power(b_y_vec,2).sum())
        return b_y_vec

    def copy_to_self(self, other):
        self.z_rad = other.z_rad
        self.x_len = other.x_len 
        self.y_len = other.y_len
        self.area = other.area
        self.center = other.center.copy()
        self.corners = other.corners.copy()
        self.R_z = other.R_z.copy() 


def min_bounding_box(pnts_in, calc_hull=True, search=0):
    """
    This is a driver function to calculate a bounding box for the 3D
    data points defined by pnts_in. The bounding box can be arbitrarily
    oriented in 3D space; the objective is to find a bounding box that
    contains all of pnts_in that also results in the minimum volume of
    said bounding box. Depending on the level of accuracy and 
    computational efficiency, different search algorithms can be used.

    ---- INPUT ARGUMENTS ----
    [[pnts_in]]: A 2D numpy array containing all of the 3D points. For n
        points, then the shape should be [n,3]. The convex hull of this 
        data set can be calculated automatically.

    calc_hull: A boolean that determines whether the convex hull should
        be calculated and used for fitting the bounding box. Setting 
        this to True often speeds up performance for larger data sets.

    search: An integer that describes what type of bounding box search
        algorithm to use. If search equals:

        0: Use an exhaustive search. Although not necessarily the
            fastest, this method should find the best bounding box 
            within 1 degree of any arbitrary rotation. This method is 
            the default search algorithm.

        1: Perform a singular-value decomposition calculation on the 
            data set and use the eigenvector corresponding to the 
            minimum variation (i.e., smallest eigenvalue) as one of the
            directions of the bounding box. The other two directions are
            found by projecting the points onto a plane normal to this
            minimum principal direction, and then solving the 2D 
            bounding box problem using an exhaustive search. In 
            practice, this method usually is slightly more accurate
            than using the maximum variation (see search=2 below) while
            being just as computationally efficient. Performing the
            convex hull usually improves the accuracy for this search.

        2: Perform a singular-value decomposition calculation on the 
            data set and use the eigenvector corresponding to the 
            maximum variation (i.e., largest eigenvalue) as one of the
            directions of the bounding box. The other two directions are
            found by projecting the points onto a plane normal to this
            maximum principal direction, and then solving the 2D 
            bounding box problem using an exhaustive search. Performing
            the convex hull usually improves the accuracy for this
            search algorithm.

        3: Perform a singular-value decomposition calculation on the 
            data set and use all three eigenvectors as directions of
            the bounding box. Although this is the fastest method for
            large data sets, this is also the least reliable method in
            terms of accuracy. Performing the convex hull usually 
            improves the accuracy for this search algorithm.

        4: Find the longest distance between two points and define this
            direction to be major z-axis of the bounding box. The other
            two directions are found by projecting the points onto a 
            plane normal to this maximum principal direction, and then
            solving the 2D bounding box problem using an exhaustive 
            search.

    ---- RETURNED ---- 
    class BBox: See above for the definition of this class, which just
        contains a number of useful attributes.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the
    hard drive. Nothing is printed to the standard output stream.
    """

    num_pnts = pnts_in.shape[0]

    # Calculate the convex hull of the data set if desired
    if num_pnts > 4:
        if calc_hull:
            hull_obj = spt.ConvexHull(pnts_in)
            hull_pnts = pnts_in[hull_obj.vertices]
        else:
            hull_pnts = pnts_in
    else:
        hull_pnts = pnts_in


    if search is None:
        bbox_min = bbox_exhaustive_search(hull_pnts)

    elif search == 0:
        bbox_min = bbox_exhaustive_search(hull_pnts)

    elif search == 1:
        # Use PCA min
        bbox_min = bbox_svd_search(hull_pnts, algorithm="MIN")

    elif search == 2:
        # Use PCA max
        bbox_min = bbox_svd_search(hull_pnts, algorithm="MAX")

    elif search == 3:
        # Use all PCA directions
        bbox_min = bbox_svd_search(hull_pnts, algorithm="ALL")

    elif search == 4:
        # Set the longest-distance pair along one axis of the bbox
        bbox_min = bbox_LWT(hull_pnts)

    else:
        print("\nWARNING: Unrecognized search ID for calculating a bounding")
        print("box in function, min_bounding_box(...). Defaulting to an")
        print("exhaustive search for minimizing volume (search = 1).")
        bbox_min = bbox_exhaustive_search(hull_pnts)

    return bbox_min


def bbox_LWT(pnts_in):

    # Number of points
    n_pnts = pnts_in.shape[0] 
    dtype_in = pnts_in.dtype

    # Using the mean point to normalize the values.
    # Will need to translate the resultant bounding box later
    mean_pnt = np.round(np.mean(pnts_in, axis=0)).astype(dtype_in)
    pnts0 = pnts_in - mean_pnt

    # Candidate points for the longest distance between a pair of
    # points will lie on the convex hull.
    #pnts0 = pnts0[spt.ConvexHull(pnts0).vertices]

    # Get distances between each pair of candidate points
    dist_mat = spt.distance_matrix(pnts0, pnts0)

    # Get indices of pnts0 that are furthest apart
    L_pnt1_i, L_pnt2_i = np.unravel_index(dist_mat.argmax(), dist_mat.shape)

    L_pnt1 = pnts0[L_pnt1_i]
    L_pnt2 = pnts0[L_pnt2_i]
    L_vec = L_pnt2 - L_pnt1
    L_mag = np.sqrt(np.power(L_vec,2).sum())
    L_hat = L_vec/L_mag

    # Rotate the points so that the L vector is parallel to the
    # z-axis
    z_hat = np.array([0, 0, 1]) # [x, y, z]
    theta_local = np.arccos(np.clip(np.dot(L_hat, z_hat), -1.0, 1.0))

    if theta_local != 0.:
        rot_axis = np.cross(L_hat, z_hat)
        pnts_rot = vol.rodrigues_rot3(pnts0, rot_axis, theta_local,
            rot_center_in=np.array([0,0,0]))
    else:
        rot_axis = np.array([1, 0, 0])
        pnts_rot = pnts0.copy()

    R_zyx = vol.make_axis_angle_rot3_matrix(rot_axis, theta_local)
    
    # the bounding box is now aligned (or rather fixed) in the 
    # Z-direction. So, now we need to project the points onto 
    # the XY-plane and solve the 2D case.
    pnts_rot_2d = pnts_rot[:,0:2] # [n, 2] array now
    bbox_2d = bbox2D_exhaustive_search(pnts_rot_2d)

    z_rad = bbox_2d.z_rad
    cz = np.cos(z_rad)
    sz = np.sin(z_rad)
    R_z = np.array([
        [cz, -sz, 0.],
        [sz, cz, 0.],
        [0., 0., 1.] ])

    # Update pnts_rot
    pnts_rot =  np.dot(R_z, np.transpose(pnts_rot))
    pnts_rot = np.transpose(pnts_rot)

    # Effectively just need to add this additional Z-rotation
    # to the existing rotation matrix: Z-Y'-X''-Z'''
    R_zyx = np.matmul(R_z, R_zyx)

    bbox_3d = BBox()

    # Store the rotation matrix
    bbox_3d.R_zyx = R_zyx

    # Calculate the z-y'-x'' Euler (actually Tait-Bryan) angles
    # https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    R21 = R_zyx[1,0]
    R11 = R_zyx[0,0]
    R31 = R_zyx[2,0]
    R32 = R_zyx[2,1]
    R33 = R_zyx[2,2]

    # Due to round-off errors, this can be technically less than 0
    temp_radicand = 1 - R31*R31
    if temp_radicand < 0.0:
        temp_radicand = 0.0

    bbox_3d.z_rad = np.arctan2(R21, R11)
    bbox_3d.y_rad = np.arctan2(-R31, np.sqrt(temp_radicand))
    bbox_3d.x_rad = np.arctan2(R32, R33)

    # Get the axis-aligned bounding box 
    min_xyz = np.amin(pnts_rot, axis=0)
    max_xyz = np.amax(pnts_rot, axis=0)
    min_x = min_xyz[0]
    max_x = max_xyz[0]
    min_y = min_xyz[1]
    max_y = max_xyz[1]
    min_z = min_xyz[2]
    max_z = max_xyz[2]

    # Storing the corners temporarily in local coordinates
    bbox_3d.corners[0,:] = np.array([max_x, min_y, min_z])
    bbox_3d.corners[1,:] = np.array([max_x, max_y, min_z])
    bbox_3d.corners[2,:] = np.array([min_x, max_y, min_z])
    bbox_3d.corners[3,:] = np.array([min_x, min_y, min_z])
    bbox_3d.corners[4,:] = np.array([max_x, min_y, max_z])
    bbox_3d.corners[5,:] = np.array([max_x, max_y, max_z])
    bbox_3d.corners[6,:] = np.array([min_x, max_y, max_z])
    bbox_3d.corners[7,:] = np.array([min_x, min_y, max_z])

    bbox_3d.x_len = max_x - min_x
    bbox_3d.y_len = max_y - min_y
    bbox_3d.z_len = max_z - min_z
    bbox_vol = bbox_3d.calc_volume()

    # Update the corner coordinates to the global coordinate system
    bbox_corners = bbox_3d.corners
    R_xyz = np.transpose(R_zyx) # Transpose is the inverse rotation

    # Note on shapes: R_xyz[3,3] * (bbox_corners[8,3])^T 
    bbox_corners2 = np.dot(R_xyz, np.transpose(bbox_corners))
    bbox_corners2 = np.transpose(bbox_corners2)

    # Make sure to translate things back based on the mean point
    bbox_3d.corners = bbox_corners2.copy() + mean_pnt

    # Use the new corner coordinates to calculate the center point
    bbox_center = bbox_3d.calc_center()

    return bbox_3d


def bbox2D_LW(pnts_in):

    # Number of points
    n_pnts = pnts_in.shape[0]
    dtype_in = pnts_in.dtype

    # Using the mean point to normalize the values.
    # Will need to translate the resultant bounding box later
    mean_pnt = np.round(np.mean(pnts_in, axis=0)).astype(dtype_in)
    pnts0 = pnts_in - mean_pnt

    # Candidate points for the longest distance between a pair of
    # points will lie on the convex hull.
    #pnts0 = pnts0[spt.ConvexHull(pnts0).vertices]

    # Get distances between each pair of candidate points
    dist_mat = spt.distance_matrix(pnts0, pnts0)

    # Get indices of pnts0 that are furthest apart
    L_pnt1_i, L_pnt2_i = np.unravel_index(dist_mat.argmax(), dist_mat.shape)

    L_pnt1 = pnts0[L_pnt1_i]
    L_pnt2 = pnts0[L_pnt2_i]
    L_vec = L_pnt2 - L_pnt1
    L_mag = np.sqrt(np.power(L_vec,2).sum())
    L_hat = L_vec/L_mag

    # Rotate the points so that the L vector is parallel to the
    # x-axis
    x_hat = np.array([1, 0]) # [x, y]

    theta_local = np.arccos(np.clip(np.dot(L_hat, x_hat), -1.0, 1.0))
    rot_axis = np.cross(L_hat, x_hat) # Only z-component returned

    if rot_axis < 0:
         theta_local = -theta_local
    
    cz = np.cos(theta_local)
    sz = np.sin(theta_local)

    R_z = np.array([
        [cz, -sz],
        [sz, cz] ])

    R_z_inv = np.array([
        [cz, sz],
        [-sz, cz] ])

    if theta_local != 0:
        # pnts0 is shape [n,2], and R_z is shape [2,2]
        pnts_rot = np.matmul(R_z, np.transpose(pnts0))

        # Convert shape from [2,n] to [n,2]
        pnts_rot = np.transpose(pnts_rot)

    else:
        pnts_rot = pnts0.copy()

    # Now find the max and min row
    min_xy = np.amin(pnts_rot, axis=0)
    max_xy = np.amax(pnts_rot, axis=0)
    min_x = min_xy[0]
    max_x = max_xy[0]
    min_y = min_xy[1]
    max_y = max_xy[1]

    # Fill in a 2D bounding box object
    bbox_2d = BBox2D()
    bbox_2d.R_z = R_z
    bbox_2d.z_rad = theta_local

    # Storing the corners temporarily in local coordinates
    bbox_2d.corners[0,:] = np.array([max_x, min_y])
    bbox_2d.corners[1,:] = np.array([max_x, max_y])
    bbox_2d.corners[2,:] = np.array([min_x, max_y])
    bbox_2d.corners[3,:] = np.array([min_x, min_y])
    bbox_corners = bbox_2d.corners

    # Worth noting that, L = max_x - min_x, 
    # and, W = max_y - min_y
    bbox_2d.x_len = max_x - min_x
    bbox_2d.y_len = max_y - min_y
    bbox_area = bbox_2d.calc_area()

    # Note on shapes: R_z_inv[2,2] * (bbox_corners[4,2])^T 
    bbox_corners2 = np.dot(R_z_inv, np.transpose(bbox_corners))
    bbox_corners2 = np.transpose(bbox_corners2)

    # Make sure to translate things back based on the mean point
    bbox_2d.corners = bbox_corners2.copy() + mean_pnt

    # Use the new corner coordinates to calculate the center point
    bbox_center = bbox_2d.calc_center()

    return bbox_2d


def bbox_exhaustive_search(pnts_in):
    """
    Calculates an arbitrarily oriented bounding box for a set of 3D data
    points. This algorithm uses a brute-force search heuristic, so for
    very large data sets (greater than 10^5 points), this may be quite
    slow. The objective is to find an oriented 3D bounding box for
    which the volume of the bounding box is minimized. For large data
    sets, you may want to consider an algorithm inspired by principal
    component analysis that uses singular value decomposition, see for
    example bbox_svd_search(...). 

    ---- INPUT ARGUMENTS ----
    [[pnts_in]]: A 2D numpy array containing all of the 3D points. For n
        points, then the shape should be [n,3].

    ---- RETURNED ---- 
    class BBox: See above for the definition of this class, which just
        contains a number of useful attributes of a bounding box. 
        Notable attributes include the corners of the box, its  
        orientation (defined by a 3D rotation matrix), the volume, 
        the center point, its length, its width, and its depth.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the
    hard drive. Nothing is printed to the standard output stream.

    """
    pnts = pnts_in.copy() # [n,3] np-array

    # Using the mean point to normalize the values.
    # Will need to translate the resultant bounding box later
    mean_pnt = np.mean(pnts, axis=0)
    pnts = pnts - mean_pnt

    # Collect some platform information for 32-bit float data type
    fi_float = np.finfo(np.single)
    vol_min = fi_float.max

    # Initialize a the minimum bounding box object
    bbox_min = BBox()

    # Create arrays of angles (in degrees) that will be tested
    # Start with a coarse search, then later, refine the search
    ang_deg_step = 5.0
    ang_i = np.arange(0.0, 90.0, ang_deg_step) # Range for Z-rotations
    ang_j = np.arange(0.0, 90.0, ang_deg_step) # Range for Y-rotations
    ang_k = np.arange(0.0, 90.0, ang_deg_step) # Range for X-rotations

    for i_deg in ang_i: # Loop through Z-rotation angles
        for j_deg in ang_j: # Loop through Y-rotation angles
            for k_deg in ang_k: # Loop through X-rotation angles

                z_rad = np.deg2rad(i_deg)
                y_rad = np.deg2rad(j_deg)
                x_rad = np.deg2rad(k_deg)

                cur_bbox = BBox(x_rad, y_rad, z_rad)
                R_zyx = cur_bbox.calc_Rzyx()

                # Rotate the convex hull points
                # Note on shapes: R_zyx[3,3] * (pnts[n,3])^T 
                pnts_rot =  np.dot(R_zyx, np.transpose(pnts))

                # Convert shape from [3,n] to [n,3]
                pnts_rot = np.transpose(pnts_rot)

                # Get the axis-aligned bounding box 
                min_xyz = np.amin(pnts_rot, axis=0)
                max_xyz = np.amax(pnts_rot, axis=0)
                min_x = min_xyz[0]
                max_x = max_xyz[0]
                min_y = min_xyz[1]
                max_y = max_xyz[1]
                min_z = min_xyz[2]
                max_z = max_xyz[2]

                # Storing the corners temporarily in local coordinates
                cur_bbox.corners[0,:] = np.array([max_x, min_y, min_z])
                cur_bbox.corners[1,:] = np.array([max_x, max_y, min_z])
                cur_bbox.corners[2,:] = np.array([min_x, max_y, min_z])
                cur_bbox.corners[3,:] = np.array([min_x, min_y, min_z])
                cur_bbox.corners[4,:] = np.array([max_x, min_y, max_z])
                cur_bbox.corners[5,:] = np.array([max_x, max_y, max_z])
                cur_bbox.corners[6,:] = np.array([min_x, max_y, max_z])
                cur_bbox.corners[7,:] = np.array([min_x, min_y, max_z])

                cur_bbox.x_len = max_x - min_x
                cur_bbox.y_len = max_y - min_y
                cur_bbox.z_len = max_z - min_z
                cur_vol = cur_bbox.calc_volume()

                # Seeking out the bounding box with smallest volume
                if cur_vol < vol_min:
                    vol_min = cur_vol
                    bbox_min.copy_to_self(cur_bbox)
                    ijk_min = (i_deg, j_deg, k_deg)

    # Now refine the search near the previously found minimum BBox
    if (ijk_min[0]-5.0) < 0:
        i_deg_left = 0
    else:
        i_deg_left = ijk_min[0]-5.0

    if (ijk_min[0]+5.0) > 90:
        i_deg_right = 90
    else:
        i_deg_right = ijk_min[0]+5.0

    if (ijk_min[1]-5.0) < 0:
        j_deg_left = 0
    else:
        j_deg_left = ijk_min[1]-5.0

    if (ijk_min[1]+5.0) > 90:
        j_deg_right = 90
    else:
        j_deg_right = ijk_min[1]+5.0

    if (ijk_min[2]-5.0) < 0:
        k_deg_left = 0
    else:
        k_deg_left = ijk_min[2]-5.0

    if (ijk_min[2]+5.0) > 90:
        k_deg_right = 90
    else:
        k_deg_right = ijk_min[2]+5.0

    ang_deg_step = 1.0
    ang_i = np.arange(i_deg_left, i_deg_right, ang_deg_step) 
    ang_j = np.arange(j_deg_left, j_deg_right, ang_deg_step) 
    ang_k = np.arange(k_deg_left, k_deg_right, ang_deg_step) 

    for i_deg in ang_i: # Loop through Z-rotation angles
        for j_deg in ang_j: # Loop through Y-rotation angles
            for k_deg in ang_k: # Loop through X-rotation angles

                z_rad = np.deg2rad(i_deg)
                y_rad = np.deg2rad(j_deg)
                x_rad = np.deg2rad(k_deg)

                cur_bbox = BBox(x_rad, y_rad, z_rad)
                R_zyx = cur_bbox.calc_Rzyx()

                # Rotate the convex hull points
                # Note on shapes: R_zyx[3,3] * (pnts[n,3])^T 
                pnts_rot =  np.dot(R_zyx, np.transpose(pnts))

                # Convert shape from [3,n] to [n,3]
                pnts_rot = np.transpose(pnts_rot)

                # Get the axis-aligned bounding box 
                min_xyz = np.amin(pnts_rot, axis=0)
                max_xyz = np.amax(pnts_rot, axis=0)
                min_x = min_xyz[0]
                max_x = max_xyz[0]
                min_y = min_xyz[1]
                max_y = max_xyz[1]
                min_z = min_xyz[2]
                max_z = max_xyz[2]

                # Storing the corners temporarily in local coordinates
                cur_bbox.corners[0,:] = np.array([max_x, min_y, min_z])
                cur_bbox.corners[1,:] = np.array([max_x, max_y, min_z])
                cur_bbox.corners[2,:] = np.array([min_x, max_y, min_z])
                cur_bbox.corners[3,:] = np.array([min_x, min_y, min_z])
                cur_bbox.corners[4,:] = np.array([max_x, min_y, max_z])
                cur_bbox.corners[5,:] = np.array([max_x, max_y, max_z])
                cur_bbox.corners[6,:] = np.array([min_x, max_y, max_z])
                cur_bbox.corners[7,:] = np.array([min_x, min_y, max_z])

                cur_bbox.x_len = max_x - min_x
                cur_bbox.y_len = max_y - min_y
                cur_bbox.z_len = max_z - min_z
                cur_vol = cur_bbox.calc_volume()

                # Seeking out the bounding box with smallest volume
                if cur_vol < vol_min:
                    vol_min = cur_vol
                    bbox_min.copy_to_self(cur_bbox)
                    ijk_min = (i_deg, j_deg, k_deg)

    # With minimum bbox found, need to update the coordinates
    bbox_corners = bbox_min.corners
    R_zyx = bbox_min.R_zyx
    R_xyz = np.transpose(R_zyx) # Transpose is the inverse rotation

    # Note on shapes: R_xyz[3,3] * (bbox_corners[8,3])^T 
    bbox_corners2 = np.dot(R_xyz, np.transpose(bbox_corners))
    bbox_corners2 = np.transpose(bbox_corners2)

    # Make sure to translate things back based on the mean point
    bbox_min.corners = bbox_corners2.copy() + mean_pnt

    # Use the new corner coordinates to calculate the center point
    bbox_center = bbox_min.calc_center()

    return bbox_min


def bbox2D_exhaustive_search(pnts_in):
    """
    Calculates an arbitrarily oriented bounding box for a set of 2D data
    points. This algorithm uses a brute-force search heuristic, so for
    very large data sets, this may be rather slow. The objective is to
    find an oriented 2D bounding box for which the area of the bounding
    box is minimized.

    ---- INPUT ARGUMENTS ----
    [[pnts_in]]: A 2D numpy array containing all of the 2D points. For n
        points, then the shape should be [n,2].

    ---- RETURNED ---- 
    class BBox2D: See above for the definition of this class, which just
        contains a number of useful attributes of a 2D bounding box. 
        Notable attributes include the corners of the box, its  
        orientation (defined by a 2D rotation matrix), the area, 
        the center point, its length, and its width.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the
    hard drive. Nothing is printed to the standard output stream.
    """

    pnts = pnts_in.copy() # [n,2] np-array

    # Using the mean point to normalize the values.
    # Will need to translate the resultant bounding box later
    mean_pnt = np.mean(pnts, axis=0)
    pnts = pnts - mean_pnt

    # Collect some platform information for 32-bit float data type
    fi_float = np.finfo(np.single)
    area_min = fi_float.max

    # Initialize a the minimum bounding box object
    bbox_min = BBox2D()

    # Create arrays of angles (in degrees) that will be tested
    # Start with a coarse search, then later, refine the search
    ang_deg_step = 5.0
    ang_i = np.arange(0.0, 90.0, ang_deg_step) # Range for Z-rotations

    for i_deg in ang_i: # Loop through Z-rotation angles

        z_rad = np.deg2rad(i_deg)

        cur_bbox = BBox2D(z_rad)
        R_z = cur_bbox.calc_Rz()

        # Rotate the convex hull points
        # Note on shapes: R_z[2,2] * (pnts[n,2])^T 
        pnts_rot =  np.dot(R_z, np.transpose(pnts))

        # Convert shape from [2,n] to [n,2]
        pnts_rot = np.transpose(pnts_rot)

        # Get the axis-aligned bounding box 
        min_xy = np.amin(pnts_rot, axis=0)
        max_xy = np.amax(pnts_rot, axis=0)
        min_x = min_xy[0]
        max_x = max_xy[0]
        min_y = min_xy[1]
        max_y = max_xy[1]

        # Storing the corners temporarily in local coordinates
        cur_bbox.corners[0,:] = np.array([max_x, min_y])
        cur_bbox.corners[1,:] = np.array([max_x, max_y])
        cur_bbox.corners[2,:] = np.array([min_x, max_y])
        cur_bbox.corners[3,:] = np.array([min_x, min_y])

        cur_bbox.x_len = max_x - min_x
        cur_bbox.y_len = max_y - min_y
        cur_area = cur_bbox.calc_area()

        # Seeking out the bounding box with smallest volume
        if cur_area < area_min:
            area_min = cur_area
            bbox_min.copy_to_self(cur_bbox)
            i_min = i_deg

    # Now refine the search near the previously found minimum BBox
    if (i_min-5.0) < 0:
        i_deg_left = 0
    else:
        i_deg_left = i_min-5.0

    if (i_min+5.0) > 90:
        i_deg_right = 90
    else:
        i_deg_right = i_min+5.0

    ang_deg_step = 1.0
    ang_i = np.arange(i_deg_left, i_deg_right, ang_deg_step) 

    for i_deg in ang_i: # Loop through Z-rotation angles

        z_rad = np.deg2rad(i_deg)

        cur_bbox = BBox2D(z_rad)
        R_z = cur_bbox.calc_Rz()

        # Rotate the convex hull points
        # Note on shapes: R_z[2,2] * (pnts[n,2])^T 
        pnts_rot =  np.dot(R_z, np.transpose(pnts))

        # Convert shape from [2,n] to [n,2]
        pnts_rot = np.transpose(pnts_rot)

        # Get the axis-aligned bounding box 
        min_xy = np.amin(pnts_rot, axis=0)
        max_xy = np.amax(pnts_rot, axis=0)
        min_x = min_xy[0]
        max_x = max_xy[0]
        min_y = min_xy[1]
        max_y = max_xy[1]

        # Storing the corners temporarily in local coordinates
        cur_bbox.corners[0,:] = np.array([max_x, min_y])
        cur_bbox.corners[1,:] = np.array([max_x, max_y])
        cur_bbox.corners[2,:] = np.array([min_x, max_y])
        cur_bbox.corners[3,:] = np.array([min_x, min_y])

        cur_bbox.x_len = max_x - min_x
        cur_bbox.y_len = max_y - min_y
        cur_area = cur_bbox.calc_area()

        # Seeking out the bounding box with smallest volume
        if cur_area < area_min:
            area_min = cur_area
            bbox_min.copy_to_self(cur_bbox)
            i_min = i_deg

    # With minimum bbox found, need to update the coordinates
    bbox_corners = bbox_min.corners
    R_z = bbox_min.R_z
    R_z_inv = np.transpose(R_z) # Transpose is the inverse rotation

    # Note on shapes: R_z_inv[2,2] * (bbox_corners[4,2])^T 
    bbox_corners2 = np.dot(R_z_inv, np.transpose(bbox_corners))
    bbox_corners2 = np.transpose(bbox_corners2)

    # Make sure to translate things back based on the mean point
    bbox_min.corners = bbox_corners2.copy() + mean_pnt

    # Use the new corner coordinates to calculate the center point
    bbox_center = bbox_min.calc_center()

    return bbox_min


def bbox_svd_search(pnts_in, algorithm="MIN"):
    """
    Utilize singular value decomposition (SVD) to calculate the singular
    values of the 3D points given by pnts_in. Some or all of the 
    resultant eigenvectors are used to define the bounding box, 
    depending on the string used to define the algorithm parameter. In
    practice, algorithm = "MIN" will usually provide a more accurate
    bounding box compared to "MAX" in terms of finding an arbitrary 
    rotation that minimizes the volume of the bounding box. However, 
    these two methods are more computationally intensive than "ALL", 
    which is the fastest and least reliably accurate method. Note, a
    bounding box that is within 1.0 deg of rotational accuracy can be 
    found using the slower, exhaustive search given by the function
    above, bbox_exhaustive_search(...).

     ---- INPUT ARGUMENTS ----
    [[pnts_in]]: A 2D numpy array containing all of the 3D points. For n
        points, then the shape should be [n,3].

    algorithm: A string that is either "MIN", "MAX", or "ALL". The 
        default value is "MIN". When set to "MIN", the local z-axis of
        the bounding box will be aligned with direction from singular
        value decomposition (SVD) that corresponds to the minimum 
        variance. When set to "MAX", the local z-axis of the bounding
        box will be aligned with the direction from SVD that corresonds
        to the maximum variance. For either "MIN" or "MAX", only one
        direction from SVD is used to define the bounding box; the 
        remaining two directions are found from solving the 2D bounding
        box problem by projecting the data to the plane normal to the
        direction kept from SVD. When set to "ALL", all three directions
        from SVD are used to define the bounding box.

    ---- RETURNED ---- 
    class BBox: See above for the definition of this class, which just
        contains a number of useful attributes of a bounding box. 
        Notable attributes include the corners of the box, its  
        orientation (defined by a 3D rotation matrix), the volume, 
        the center point, its length, its width, and its depth.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the
    hard drive. Nothing is printed to the standard output stream.
    """
    
    pnts = pnts_in.copy()
    alg = algorithm.upper() # MIN, MAX, or ALL

    # Using the mean point to normalize the values.
    # Will need to translate the resultant bounding box later
    mean_pnt = np.mean(pnts, axis=0)
    pnts = pnts - mean_pnt

    # Use SVD to calculate the principal directions
    #  pnts: [n, 3]
    #  u: [n, n]
    #  s: [3]   <--- Singular values sorted in descending order
    #  vh: [3, 3]
    # The rows of vh are the eigenvectors of pnts^T * pnts
    # The columns of u are the eigenvectors of pnts * pnts^T
    u, s, vh = np.linalg.svd(pnts)

    # Note, the following code calculates pnts via the equation:
    #    pnts = u*s_mat*vh
    #
    # ---- Uncomment Below To Activate Code -----
    #
    #s_mat = np.zeros( (u.shape[0], vh.shape[0]) )
    #for m, cur_s in enumerate(s):
    #    s_mat[m,m] = cur_s
    #    
    #pnts_copy = np.matmul(u, np.matmul(s_mat, vh) )

    eig_vec1 = vh[0,:]
    eig_vec2 = vh[1,:]
    eig_vec3 = vh[2,:]

    mag1 = np.sqrt( np.power(eig_vec1[0], 2) + 
                    np.power(eig_vec1[1], 2) +
                    np.power(eig_vec1[2], 2) )
    eig_vec1 = eig_vec1/mag1

    mag2 = np.sqrt( np.power(eig_vec2[0], 2) + 
                    np.power(eig_vec2[1], 2) +
                    np.power(eig_vec2[2], 2) )
    eig_vec2 = eig_vec2/mag2

    mag3 = np.sqrt( np.power(eig_vec3[0], 2) + 
                    np.power(eig_vec3[1], 2) +
                    np.power(eig_vec3[2], 2) )
    eig_vec3 = eig_vec3/mag3

    if alg == "ALL":
        # Let us assume the bounding box will align its local x'-axis
        # to eig_vec1, y'-axis to eig_vec2, and z'-axis to eig_vec3.

        # The bbox x-vector expressed in global coordinates
        x_vec_g = eig_vec1 

        # The bbox y-vector expressed in global coordinates
        y_vec_g = eig_vec2

        # The bbox z-vector expressed in global coordinates
        z_vec_g = np.cross(x_vec_g, y_vec_g)  

    elif alg == "MAX":
        # In this case, the bounding box will align its local z'-axis
        # to eig_vec1. The remaining local axes can be arbitrarily 
        # chosen for the time being.

        # The bbox z-vector expressed in global coordinates
        z_vec_g = eig_vec1 

        # The bbox y-vector expressed in global coordinates
        y_vec_g = eig_vec2

        # The bbox x-vector expressed in global coordinates
        x_vec_g = np.cross(y_vec_g, z_vec_g)

    else: # "MIN"
        # In this case, the bounding box will align its local z'-axis
        # to eig_vec3. The remaining local axes can be arbitrarily 
        # chosen for the time being.

        # The bbox z-vector expressed in global coordinates
        z_vec_g = eig_vec3 

        # The bbox y-vector expressed in global coordinates
        y_vec_g = eig_vec2

        # The bbox x-vector expressed in global coordinates
        x_vec_g = np.cross(y_vec_g, z_vec_g)

    # Make the rotation matrix
    R_zyx = np.array([
        [x_vec_g[0], x_vec_g[1], x_vec_g[2] ],
        [y_vec_g[0], y_vec_g[1], y_vec_g[2] ],
        [z_vec_g[0], z_vec_g[1], z_vec_g[2] ] ])

    # Rotate the points and work in the local coordinate system
    # Note on shapes: R_zyx[3,3] * (pnts[n,3])^T 
    pnts_rot =  np.dot(R_zyx, np.transpose(pnts))

    # Convert shape from [3,n] to [n,3]
    pnts_rot = np.transpose(pnts_rot)

    if alg != "ALL":
        # For either the "MIN" or "MAX" case, the bounding box is now
        # aligned (or rather fixed) in the Z-direction. So, now we need
        # to project the points onto the XY-plane and solve the 2D case
        pnts_rot_2d = pnts_rot[:,0:2] # [n, 2] array now
        bbox_2d = bbox2D_exhaustive_search(pnts_rot_2d)

        z_rad = bbox_2d.z_rad
        cz = np.cos(z_rad)
        sz = np.sin(z_rad)
        R_z = np.array([
            [cz, -sz, 0.],
            [sz, cz, 0.],
            [0., 0., 1.] ])

        # Update pnts_rot
        pnts_rot =  np.dot(R_z, np.transpose(pnts_rot))
        pnts_rot = np.transpose(pnts_rot)

        # Effectively just need to add this additional Z-rotation
        # to the existing rotation matrix: Z-Y'-X''-Z'''
        R_zyx = np.matmul(R_z, R_zyx)

    # Initialize the BBox object that will be returned
    bbox_min = BBox()

    # Store the rotation matrix
    bbox_min.R_zyx = R_zyx

    # Calculate the z-y'-x'' Euler (actually Tait-Bryan) angles
    # https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    R21 = R_zyx[1,0]
    R11 = R_zyx[0,0]
    R31 = R_zyx[2,0]
    R32 = R_zyx[2,1]
    R33 = R_zyx[2,2]

    # Due to round-off errors, this can be technically less than 0
    temp_radicand = 1 - R31*R31
    if temp_radicand < 0.0:
        temp_radicand = 0.0

    bbox_min.z_rad = np.arctan2(R21, R11)
    bbox_min.y_rad = np.arctan2(-R31, np.sqrt(temp_radicand))
    bbox_min.x_rad = np.arctan2(R32, R33)

    # Get the axis-aligned bounding box 
    min_xyz = np.amin(pnts_rot, axis=0)
    max_xyz = np.amax(pnts_rot, axis=0)
    min_x = min_xyz[0]
    max_x = max_xyz[0]
    min_y = min_xyz[1]
    max_y = max_xyz[1]
    min_z = min_xyz[2]
    max_z = max_xyz[2]

    # Storing the corners temporarily in local coordinates
    bbox_min.corners[0,:] = np.array([max_x, min_y, min_z])
    bbox_min.corners[1,:] = np.array([max_x, max_y, min_z])
    bbox_min.corners[2,:] = np.array([min_x, max_y, min_z])
    bbox_min.corners[3,:] = np.array([min_x, min_y, min_z])
    bbox_min.corners[4,:] = np.array([max_x, min_y, max_z])
    bbox_min.corners[5,:] = np.array([max_x, max_y, max_z])
    bbox_min.corners[6,:] = np.array([min_x, max_y, max_z])
    bbox_min.corners[7,:] = np.array([min_x, min_y, max_z])

    bbox_min.x_len = max_x - min_x
    bbox_min.y_len = max_y - min_y
    bbox_min.z_len = max_z - min_z
    cur_vol = bbox_min.calc_volume()

    # Axis-aligned local bbox found; need to update the corner 
    # coordinates to the global coordinate system
    bbox_corners = bbox_min.corners
    R_xyz = np.transpose(R_zyx) # Transpose is the inverse rotation

    # Note on shapes: R_xyz[3,3] * (bbox_corners[8,3])^T 
    bbox_corners2 = np.dot(R_xyz, np.transpose(bbox_corners))
    bbox_corners2 = np.transpose(bbox_corners2)

    # Make sure to translate things back based on the mean point
    bbox_min.corners = bbox_corners2.copy() + mean_pnt

    # Use the new corner coordinates to calculate the center point
    bbox_center = bbox_min.calc_center()

    return bbox_min