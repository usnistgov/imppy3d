import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

cimport cython

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE1 for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE1 = np.uint8
DTYPE2 = np.uint32
DTYPE3 = np.float32
DTYPE4 = np.int32
DTYPE5 = np.uint64

# "ctypedef" assigns a corresponding compile-time type to DTYPE1_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
# https://numpy.org/doc/1.19/user/basics.types.html?highlight=dtypes
ctypedef np.uint8_t DTYPE1_t
ctypedef np.uint32_t DTYPE2_t
ctypedef np.float32_t DTYPE3_t
ctypedef np.int32_t DTYPE4_t
ctypedef np.uint64_t DTYPE5_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

# "def" can type its arguments but not have a return type. The type of the
# arguments for a "def" function is checked at run-time when entering the
# function
def get_voxel_info(np.ndarray[DTYPE1_t, ndim=3] img_arr, DTYPE4_t debug_flag):

    # Some basic checks for the input arrays
    cdef DTYPE2_t img_dim = img_arr.ndim
    if img_dim != 3:
        print(f"Number of dimensions in img_arr: {img_dim}")
        raise ValueError("img_arr must be a 3D Numpy array")
    assert img_arr.dtype == DTYPE1

    # Constants to be used later
    cdef DTYPE1_t ONE_UINT8 = 1
    cdef DTYPE2_t ONE_UINT32 = 1
    cdef Py_ssize_t ONE_PYSZT = 1
    cdef DTYPE3_t HALF = 0.5
    cdef DTYPE2_t MAX_UINT32 = 4294967295
    cdef DTYPE2_t EIGHT = 8

    # Py_ssize_t is essentially just an int (for the Python purist) which
    # should be used for index in Cython. Can actually use an int here 
    # and it would be fine
    cdef DTYPE2_t n_imgs = img_arr.shape[0]
    cdef DTYPE2_t n_rows = img_arr.shape[1]
    cdef DTYPE2_t n_cols = img_arr.shape[2]
    cdef Py_ssize_t ii, rr, cc, xc, yc, zc # For-loop indices

    # The number of voxels that will be created. Initialize the Numpy
    # connectivity array that will be used to create the VTK model
    cdef DTYPE2_t num_cells = 0
    cdef DTYPE2_t num_verts = 0
    cdef DTYPE1_t temp_UINT8

    if debug_flag > 0:
        print(f"\nTemporarily allocating "\
            f"{(n_imgs+1)*(n_rows+1)*(n_cols+1)*4/1E6} MB of memory")

    cdef np.ndarray[DTYPE2_t, ndim=3] vert_ids = np.ones(
        (n_imgs+1, n_rows+1, n_cols+1), dtype=DTYPE2)

    cdef np.ndarray[DTYPE3_t, ndim=1] nd1 = np.zeros(3, dtype=DTYPE3)
    cdef np.ndarray[DTYPE3_t, ndim=1] nd2 = np.zeros(3, dtype=DTYPE3)
    cdef np.ndarray[DTYPE3_t, ndim=1] nd3 = np.zeros(3, dtype=DTYPE3)
    cdef np.ndarray[DTYPE3_t, ndim=1] nd4 = np.zeros(3, dtype=DTYPE3)
    cdef np.ndarray[DTYPE3_t, ndim=1] nd5 = np.zeros(3, dtype=DTYPE3)
    cdef np.ndarray[DTYPE3_t, ndim=1] nd6 = np.zeros(3, dtype=DTYPE3)
    cdef np.ndarray[DTYPE3_t, ndim=1] nd7 = np.zeros(3, dtype=DTYPE3)
    cdef np.ndarray[DTYPE3_t, ndim=1] nd8 = np.zeros(3, dtype=DTYPE3)

    cdef DTYPE2_t nd1_id, nd2_id, nd3_id, nd4_id, nd5_id, nd6_id, nd7_id, nd8_id

    # Initialize the vert_ids 3D array to the largest possible value that
    # can be stored in a UINT32. Any vertex with a corresponding value in
    # this array equal to this maximum value will not be used in the final
    # voxel model.
    if debug_flag > 0:
        print(f"\nCalculating additional memory allocation requirements...")

    for ii in range(n_imgs+ONE_UINT32):
        for rr in range(n_rows+ONE_UINT32):
            for cc in range(n_cols+ONE_UINT32):
                vert_ids[ii,rr,cc] = MAX_UINT32

    # First, loop through the image sequence to find out how many vertices and
    # cells will actually be needed. This triple for-loop is just to allocate 
    # the vertex and cell arrays to the correct size later.
    for ii in range(n_imgs):
        for rr in range(n_rows):
            for cc in range(n_cols):

                temp_UINT8 = img_arr[ii,rr,cc]
                if temp_UINT8 < ONE_UINT8:
                    continue # Skip black voxels

                num_cells = num_cells + ONE_UINT32

                # Node 1
                if vert_ids[ii, rr, cc] == MAX_UINT32:
                    vert_ids[ii, rr, cc] = num_verts
                    num_verts = num_verts + ONE_UINT32

                # Node 2
                if vert_ids[ii, rr, cc+ONE_PYSZT] == MAX_UINT32:
                    vert_ids[ii, rr, cc+ONE_PYSZT] = num_verts
                    num_verts = num_verts + ONE_UINT32

                # Node 3
                if vert_ids[ii, rr+ONE_PYSZT, cc+ONE_PYSZT] == MAX_UINT32:
                    vert_ids[ii, rr+ONE_PYSZT, cc+ONE_PYSZT] = num_verts
                    num_verts = num_verts + ONE_UINT32

                # Node 4
                if vert_ids[ii, rr+ONE_PYSZT, cc] == MAX_UINT32:
                    vert_ids[ii, rr+ONE_PYSZT, cc] = num_verts
                    num_verts = num_verts + ONE_UINT32

                # Node 5
                if vert_ids[ii+ONE_PYSZT, rr, cc] == MAX_UINT32:
                    vert_ids[ii+ONE_PYSZT, rr, cc] = num_verts
                    num_verts = num_verts + ONE_UINT32

                # Node 6
                if vert_ids[ii+ONE_PYSZT, rr, cc+ONE_PYSZT] == MAX_UINT32:
                    vert_ids[ii+ONE_PYSZT, rr, cc+ONE_PYSZT] = num_verts
                    num_verts = num_verts + ONE_UINT32

                # Node 7
                if vert_ids[ii+ONE_PYSZT, rr+ONE_PYSZT, cc+ONE_PYSZT] == MAX_UINT32:
                    vert_ids[ii+ONE_PYSZT, rr+ONE_PYSZT, cc+ONE_PYSZT] = num_verts
                    num_verts = num_verts + ONE_UINT32

                # Node 8
                if vert_ids[ii+ONE_PYSZT, rr+ONE_PYSZT, cc] == MAX_UINT32:
                    vert_ids[ii+ONE_PYSZT, rr+ONE_PYSZT, cc] = num_verts
                    num_verts = num_verts + ONE_UINT32

    print(f"  Found {num_verts} vertices")
    print(f"  Found {num_cells} voxels")

    if debug_flag > 0:
        print(f"\nAllocating approximately "\
            f"{(num_verts*3 + num_cells*9)*4/1E6 + num_cells/1E6}"\
            f" MB of additional memory")

    cdef np.ndarray[DTYPE3_t, ndim=2] nd_coords = np.zeros(
        (num_verts,3), dtype=DTYPE3)

    cdef np.ndarray[DTYPE2_t, ndim=2] cell_conn = np.zeros(
        (num_cells,9), dtype=DTYPE2)

    cdef np.ndarray[DTYPE1_t, ndim=1] cell_vals = np.zeros(
        num_cells, dtype=DTYPE1)

    if (num_verts == 0) or (num_cells == 0):
        return [nd_coords, cell_conn, cell_vals]

    # Now that everything is allocated, re-initialize the vertex array
    print(f"\nBuilding voxel connectivity matrix...")
    for ii in range(n_imgs+ONE_UINT32):
        for rr in range(n_rows+ONE_UINT32):
            for cc in range(n_cols+ONE_UINT32):
                vert_ids[ii,rr,cc] = MAX_UINT32

    # This triple for-loop is the one that will actually store the nodal
    # coordinates of each vertex, and construct the connectivity matrix.
    # That is, the nodal indices (or Node IDs) used to define each cell 
    # (or voxel).
    num_verts = 0 # Also using these as indices, not just counters
    num_cells = 0 # Also using these as indices, not just counters
    for ii in range(n_imgs):
        for rr in range(n_rows):
            for cc in range(n_cols):

                temp_UINT8 = img_arr[ii,rr,cc]
                if temp_UINT8 < ONE_UINT8:
                    continue # Skip black voxels

                xc = cc # Switching the indices around!
                yc = rr
                zc = ii
                
                # Calculate vertex/node coordinates for each cell
                nd1[0] = xc - HALF
                nd1[1] = yc - HALF
                nd1[2] = zc - HALF
                
                nd2[0] = xc + HALF
                nd2[1] = yc - HALF
                nd2[2] = zc - HALF
                
                nd3[0] = xc + HALF
                nd3[1] = yc + HALF
                nd3[2] = zc - HALF
                
                nd4[0] = xc - HALF
                nd4[1] = yc + HALF
                nd4[2] = zc - HALF
                
                nd5[0] = xc - HALF
                nd5[1] = yc - HALF
                nd5[2] = zc + HALF
                
                nd6[0] = xc + HALF
                nd6[1] = yc - HALF
                nd6[2] = zc + HALF
                
                nd7[0] = xc + HALF
                nd7[1] = yc + HALF
                nd7[2] = zc + HALF
                
                nd8[0] = xc - HALF
                nd8[1] = yc + HALF
                nd8[2] = zc + HALF

                # Determine the node IDs for each cell. This requires
                # checking to see if the node already has been activated
                # in the vert_ids 3D array. If it has not, then activate 
                # it and store the new node coordinates into nd_coords.
                # If it has, then just need the node ID. Do this for each
                # node.

                # Node 1
                if vert_ids[ii, rr, cc] == MAX_UINT32:
                    vert_ids[ii, rr, cc] = num_verts 
                    nd1_id = num_verts # New node/vertex
                    nd_coords[num_verts,0] = nd1[0]
                    nd_coords[num_verts,1] = nd1[1]
                    nd_coords[num_verts,2] = nd1[2]
                    num_verts = num_verts + ONE_UINT32
                else:
                    # Existing node/vertex
                    nd1_id = vert_ids[ii, rr, cc] 

                # Node 2
                if vert_ids[ii, rr, cc+ONE_PYSZT] == MAX_UINT32:
                    vert_ids[ii, rr, cc+ONE_PYSZT] = num_verts 
                    nd2_id = num_verts # New node/vertex
                    nd_coords[num_verts,0] = nd2[0]
                    nd_coords[num_verts,1] = nd2[1]
                    nd_coords[num_verts,2] = nd2[2]
                    num_verts = num_verts + ONE_UINT32
                else:
                    # Existing node/vertex
                    nd2_id = vert_ids[ii, rr, cc+ONE_PYSZT]

                # Node 3
                if vert_ids[ii, rr+ONE_PYSZT, cc+ONE_PYSZT] == MAX_UINT32:
                    vert_ids[ii, rr+ONE_PYSZT, cc+ONE_PYSZT] = num_verts 
                    nd3_id = num_verts # New node/vertex
                    nd_coords[num_verts,0] = nd3[0]
                    nd_coords[num_verts,1] = nd3[1]
                    nd_coords[num_verts,2] = nd3[2]
                    num_verts = num_verts + ONE_UINT32
                else:
                    # Existing node/vertex
                    nd3_id = vert_ids[ii, rr+ONE_PYSZT, cc+ONE_PYSZT]

                # Node 4
                if vert_ids[ii, rr+ONE_PYSZT, cc] == MAX_UINT32:
                    vert_ids[ii, rr+ONE_PYSZT, cc] = num_verts
                    nd4_id = num_verts # New node/vertex
                    nd_coords[num_verts,0] = nd4[0]
                    nd_coords[num_verts,1] = nd4[1]
                    nd_coords[num_verts,2] = nd4[2]
                    num_verts = num_verts + ONE_UINT32
                else:
                    # Existing node/vertex
                    nd4_id = vert_ids[ii, rr+ONE_PYSZT, cc]

                # Node 5
                if vert_ids[ii+ONE_PYSZT, rr, cc] == MAX_UINT32:
                    vert_ids[ii+ONE_PYSZT, rr, cc] = num_verts
                    nd5_id = num_verts # New node/vertex
                    nd_coords[num_verts,0] = nd5[0]
                    nd_coords[num_verts,1] = nd5[1]
                    nd_coords[num_verts,2] = nd5[2]
                    num_verts = num_verts + ONE_UINT32
                else:
                    # Existing node/vertex
                    nd5_id = vert_ids[ii+ONE_PYSZT, rr, cc]

                # Node 6
                if vert_ids[ii+ONE_PYSZT, rr, cc+ONE_PYSZT] == MAX_UINT32:
                    vert_ids[ii+ONE_PYSZT, rr, cc+ONE_PYSZT] = num_verts
                    nd6_id = num_verts # New node/vertex
                    nd_coords[num_verts,0] = nd6[0]
                    nd_coords[num_verts,1] = nd6[1]
                    nd_coords[num_verts,2] = nd6[2]
                    num_verts = num_verts + ONE_UINT32
                else:
                    # Existing node/vertex
                    nd6_id = vert_ids[ii+ONE_PYSZT, rr, cc+ONE_PYSZT]

                # Node 7
                if vert_ids[ii+ONE_PYSZT, rr+ONE_PYSZT, cc+ONE_PYSZT] == MAX_UINT32:
                    vert_ids[ii+ONE_PYSZT, rr+ONE_PYSZT, cc+ONE_PYSZT] = num_verts
                    nd7_id = num_verts # New node/vertex
                    nd_coords[num_verts,0] = nd7[0]
                    nd_coords[num_verts,1] = nd7[1]
                    nd_coords[num_verts,2] = nd7[2]
                    num_verts = num_verts + ONE_UINT32
                else:
                    # Existing node/vertex
                    nd7_id = vert_ids[ii+ONE_PYSZT, rr+ONE_PYSZT, cc+ONE_PYSZT]

                # Node 8
                if vert_ids[ii+ONE_PYSZT, rr+ONE_PYSZT, cc] == MAX_UINT32:
                    vert_ids[ii+ONE_PYSZT, rr+ONE_PYSZT, cc] = num_verts
                    nd8_id = num_verts # New node/vertex
                    nd_coords[num_verts,0] = nd8[0]
                    nd_coords[num_verts,1] = nd8[1]
                    nd_coords[num_verts,2] = nd8[2]
                    num_verts = num_verts + ONE_UINT32
                else:
                    # Existing node/vertex
                    nd8_id = vert_ids[ii+ONE_PYSZT, rr+ONE_PYSZT, cc]

                # Store each node ID into the cell array. I call this the
                # connectivity matrix. Note, each row starts with the number
                # eight, then the node IDs are stored. This is a VTK thing,
                # and the number eight tells VTK to look for eight node IDs
                # and that this is a definition of a hexahedron.
                cell_conn[num_cells,0] = EIGHT
                cell_conn[num_cells,1] = nd1_id
                cell_conn[num_cells,2] = nd2_id
                cell_conn[num_cells,3] = nd3_id
                cell_conn[num_cells,4] = nd4_id
                cell_conn[num_cells,5] = nd5_id
                cell_conn[num_cells,6] = nd6_id
                cell_conn[num_cells,7] = nd7_id
                cell_conn[num_cells,8] = nd8_id

                # Store the intensity value for this voxel too
                cell_vals[num_cells] = temp_UINT8

                num_cells = num_cells + ONE_UINT32

    print(f"  Success!")
    return [nd_coords, cell_conn, cell_vals]


def extract_sphere(np.ndarray[DTYPE1_t, ndim=3] img_arr, DTYPE2_t radius):

    # Some basic checks for the input arrays
    cdef DTYPE2_t img_dim = img_arr.ndim

    if img_dim != 3:
        print(f"Number of dimensions in image array: {img_dim}")
        raise ValueError("Image array, img_arr, must be a 3D Numpy array")
    assert img_arr.dtype == DTYPE1

    # Py_ssize_t is essentially just an int (for the Python purist) which
    # should be used for index in Cython. Can actually use an int here 
    # and it would be fine
    cdef DTYPE2_t n_imgs = img_arr.shape[0]
    cdef DTYPE2_t n_rows = img_arr.shape[1]
    cdef DTYPE2_t n_cols = img_arr.shape[2]
    cdef DTYPE2_t n_i_half = (np.floor(n_imgs/2)).astype(DTYPE2)
    cdef DTYPE2_t n_r_half = (np.floor(n_rows/2)).astype(DTYPE2)
    cdef DTYPE2_t n_c_half = (np.floor(n_cols/2)).astype(DTYPE2)

    if (n_i_half-1) < radius:
        print(f"\nNumber of image slices in image array: {n_imgs}")
        print(f"Desired spherical radius: {radius}")
        print(f"WARNING: Radius is not less than half of"+\
            " the number of image slices in image array")
        print(f"Setting spherical radius to {n_i_half-1}\n")
        radius = n_i_half-1

    if (n_r_half-1) < radius:
        print(f"\nNumber of rows in image array: {n_rows}")
        print(f"Desired spherical radius: {radius}")
        print(f"WARNING: Radius is not less than half of"+\
            " the number of rows in image array")
        print(f"Setting spherical radius to {n_r_half-1}\n")
        radius = n_r_half-1

    if (n_c_half-1) < radius:
        print(f"\nNumber of columns in image array: {n_cols}")
        print(f"Desired spherical radius: {radius}")
        print(f"WARNING: Radius is not less than half of"+\
            " the number of columns in image array")
        print(f"Setting spherical radius to {n_c_half-1}\n")
        radius = n_c_half-1

    cdef Py_ssize_t ii, rr, cc, x, y, z,
    cdef DTYPE2_t a, b, c

    # The extracted sphere of voxels will be saved in this image 
    # sequence and returned
    cdef np.ndarray[DTYPE1_t, ndim=3] img_sph = np.zeros(
        (n_imgs, n_rows, n_cols), dtype=DTYPE1)

    cdef DTYPE5_t r_sqrd, lhs

    a = n_c_half # X center point
    b = n_r_half # Y center point
    c = n_i_half # Z center point
    r_sqrd = radius*radius

    # Loop through image sequence to find all of the voxels within a 
    # spherical radius from the center coordinate
    for ii in range(n_imgs):
        for rr in range(n_rows):
            for cc in range(n_cols): 

                x = cc
                y = rr
                z = ii

                # Sphere equation
                # (x-a)^2 + (y-b)^2 + (z-c)^2 = r^2
                lhs = (x-a)*(x-a) + (y-b)*(y-b) + (z-c)*(z-c)

                if lhs <= r_sqrd:
                    img_sph[ii,rr,cc] = img_arr[ii,rr,cc]

    return img_sph

