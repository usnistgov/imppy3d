import numpy as np
from scipy import ndimage as ndim
import vtk
import cython.bin.im3_processing as im3
import pyvista as pv
from skimage import measure as meas
from skimage.util import img_as_ubyte
from skimage.util import img_as_float32

import volume_image_processing as vol


def make_vtk_uniform_grid(img_arr_in,  scale_spacing=1.0, 
    output_rect_grid=False):
    """
    Create a vtkImageData class from a 3D image sequence using PyVista's
    UniformGrid wrapper. Use PyVista UniformGrid plot() function to 
    visualize the uniform grid model in 3D. To save it to a VTK file,
    use the PyVista save() function. Note, the intensity values will
    be saved in the uniform grid under the name "values".

    ---- INPUT ARGUMENTS ---- 
    [[[img_arr_in]]]: A 3D Numpy array representing the image sequence
        It is important that this is a Numpy array and not a Python list
        of Numpy matrices. The shape of img_arr_in is expected to be as
        (num_images, num_pixel_rows, num_pixel_cols). Note, when
        converting this to spatial coordinates, num_images is the
        Z-component, num_pixel_rows is the Y-component, and
        num_pixel_cols is the X-component. It is expected that the
        images are single-channel, and the data should be of type uint8.
        The intensity values will be assigned to each voxel in the 
        structured grid, so grayscale intensities could be used or a
        binarized image sequence could be used.

    scale_spacing: A scalar float that represents the distance between
        the voxels before they are converted to a surface. If this is
        an X-ray CT image sequence, this represents the voxel size.

    output_rect_grid: A boolean to determine whether a 
        pyvista.RectilinearGrid object should be returned instead of
        a pyvista.UniformGrid object. The pyvista.RectilinearGrid object
        is a wrapper for a vtk.vtkRectilinearGrid object.

    ---- RETURNED ---- 
    uni_grid: A pyvista.UniformGrid object is returned. PyVista is a 
        Python library that acts as a high-level wrapper for VTK - the
        visualization toolkit. PyVista's UniformGrid object is an 
        extension of VTK's vtkImageData class. Note, if output_rect_grid
        is True, then a pyvista.RectilinearGrid object is returned. A
        pyvista.RectilinearGrid object is wrapper for a 
        vtk.vtkRectilinearGrid object. For more information, see
        https://docs.pyvista.org/core/grids.html

    ---- SIDE EFFECTS ---- 
    Function input arguments should not be altered. However, a new view
    is created for img_arr_in, not a deep copy. So, this cannot be 
    guaranteed. Nothing is written to the hard drive.
    """

    img_arr = np.transpose(img_arr_in) # New view, but not deep copy

    # Create the spatial reference
    uni_grid = pv.UniformGrid()

    # Set the grid dimensions: shape + 1 because we want to inject the values on
    # the CELL data
    uni_grid.dimensions = np.array(img_arr.shape) + 1

    # Translate the origin of the mesh
    #grid.origin = (100, 33, 55.6)

    # These are the cell sizes along each axis. Uniform/unity spacing
    uni_grid.spacing = (scale_spacing, scale_spacing, scale_spacing)  

    # Add the data values to the cell data
    # Flatten the array (in Fortran style)
    uni_grid.cell_data["values"] = img_arr.flatten(order="F") 

    if output_rect_grid:
        uni_grid = uni_grid.cast_to_rectilinear_grid()

    return uni_grid


def convert_voxels_to_surface(img_arr_in, iso_level=125, scale_spacing=1.0,
    is_binary=True, g_sigdev=0.8, pad_boundary=True):
    """
    Applies a marching cubes algorithm to an image sequence (i.e., a
    voxel model) in order to calculate an isosurface. The algorithm used
    is the SciKit-Image algorithm, skimage.measure.marching_cubes(). It
    is based on the Lewiner et al. approach:

    Thomas Lewiner, Helio Lopes, Antonio Wilson Vieira and Geovan Tavares.
      Efficient implementation of Marching Cubes’ cases with topological 
      guarantees. Journal of Graphics Tools 8(2) pp. 1-15 (December 2003).
      DOI:10.1080/10867651.2003.10487582

    No additional smoothing or modification to the resultant mesh is 
    performed. If this is desired, then consider making a VTK surface
    mesh and using Laplacian smoothing, which is all available with 
    little effort in make_vtk_surf_mesh() defined below. To measure
    the surface area, use skimage.measure.mesh_surface_area(). Or, 
    create a VTK surface mesh using make_vtk_surf_mesh() and then
    retrieve the pyvista.PolyData.area property.

    ---- INPUT ARGUMENTS ---- 
    [[[img_arr_in]]]: A 3D Numpy array representing the image sequence
        It is important that this is a Numpy array and not a Python list
        of Numpy matrices. The shape of img_arr_in is expected to be as
        (num_images, num_pixel_rows, num_pixel_cols). Note, when
        converting this to spatial coordinates, num_images is the
        Z-component, num_pixel_rows is the Y-component, and
        num_pixel_cols is the X-component. It is expected that the
        images are single-channel, and the data should be of type uint8.
        Specifically, intensities should range from 0 to 255. If it is
        segmented (i.e., binarized), then only values of 0 and 255 
        should be present (NOT 0 and 1).

    iso_level: A scalar number used to define the value for the iso-
        surface. For a binarized image sequence, a value in between 
        0 and 255, like 125, is recommended. For grayscale image
        sequence, any value can be used, and the corresponding 
        isosurface will be approximated.

    scale_spacing: A scalar float that represents the distance between
        the voxels before they are converted to a surface. If this is
        an X-ray CT image sequence, this represents the voxel size.

    is_binary: Set to True if the image sequence is binarized. That is,
        if it contains only values of 0 and 255. Using the binaraized
        image sequence directly into the marching cubes algorithm will
        result in a very poor isosurface. So, if is_binary is True, 
        then a small amount of Gaussian blurring is performed which makes 
        for a much smoother gradient between black and white. This
        vastly improves the accuracy of the marching cubes algorithm.

    g_sigdev: A single float that corresponds to the standard deviation
        used as input for the 3D Gaussian blur that must be applied for
        a binary image sequence. If is_binary is False, this value is
        ignored. The same standard deviation will be used for all three
        axes. Typical values range between 0.5 and 1.5. This blur helps
        to stabilize the marching cubes algorithm, but it also has the
        side effect of smoothing edges by a small amount.

    pad_boundary: Set to True to extend the boundaries of the image
        sequence in all three dimensions. More specifically, in all
        six directions of the rectangular volume defined by the image
        sequence. The six boundaries will be extended by some amount. 
        The motivation to do this is to ensure
        there are no white pixels along the boundaries of the image
        sequence. Otherwise, the resultant surface will not be a 
        closed at the boundaries. If there are no white pixels along the
        boundaries to begin with, then there is no need for padding.
        
    ---- RETURNED ----
    [[verts]]: A 2D array that contains the coordinates of the mesh 
        vertices. The array will have 3 columns, such that each row
        will provide the [X, Y, Z] coordinates of that vertex. There
        will be V unique mesh vertices. So, the size is (V, 3).

    [[faces]]: A 2D array that defines the connectivity of the 
        triangular faces. There will be F faces, and for each face,
        three nodes are defined based on the indices of verts. So, the
        size is (F, 3).

    [[normals]]: A 2D array that defines the normal vectors at each
        vertex. So, the size is (V, 3). Each normal vector contains
        a [X, Y, Z] components.

    [values]: A 1D array containing the approximate pixel values
        at each vertex. So, the size is (1, V)

    ---- SIDE EFFECTS ---- 
    Function input arguments should not be altered. However, a new view
    is created for img_arr_in, not a deep copy. So, this cannot be 
    guaranteed. Nothing is written to the hard drive.
    """

    img_arr = np.transpose(img_arr_in) # New view, but not deep copy

    if is_binary:
        # Blurring something white at the edge does not create a 
        # smooth gradient to black. So, pad the edges with black
        # pixels by making the image sequence larger.
        img_arr = vol.pad_image_boundary(img_arr, n_pad_in=9, quiet_in=True)

        # Apply a little blur to get better isosurfaces on a binary
        # image sequence. Need smoother gradients
        g_sigma = (g_sigdev, g_sigdev, g_sigdev) 
        img_arr = img_as_float32(img_arr)
        img_arr = ndim.gaussian_filter(img_arr, g_sigma)
        img_arr = img_as_ubyte(img_arr)

    if pad_boundary:
        # If an isosurface comes to the edge, the resultant mesh is left
        # with an open hole. By padding the image array with black pixels
        # (by making it a little larger), the isosurface will be closed.
        img_arr = vol.pad_image_boundary(img_arr, n_pad_in=1, quiet_in=True)

    # Apply marching cubes algorithm
    # verts is a (V,3) array. 
    # faces is a (F,3) array.
    # normals is a (V,3) array.
    # values is a (V,) array
    verts, faces, normals, values = meas.marching_cubes(img_arr, 
        level=iso_level, spacing=(scale_spacing, scale_spacing, scale_spacing), 
        gradient_direction='descent', step_size=1, 
        allow_degenerate=False, method='lewiner')

    # Offset the vertices by the amount of padding
    if is_binary:
        verts[:,0] = verts[:,0] - 9*scale_spacing
        verts[:,1] = verts[:,1] - 9*scale_spacing
        verts[:,2] = verts[:,2] - 9*scale_spacing

    if pad_boundary:
        verts[:,0] = verts[:,0] - 1*scale_spacing
        verts[:,1] = verts[:,1] - 1*scale_spacing
        verts[:,2] = verts[:,2] - 1*scale_spacing

    return [verts, faces, normals, values]


def make_vtk_surf_mesh(verts_in, faces_in, values_in, smth_iter=5):
    """
    Creates a PyVista PolyData object, which is an extension of the
    vtk.vtkPolyData object. Based on a set of nodal coordinates and
    the connectivity of these nodes to form triangular faces, and 
    surface mesh object is created. Use PyVista PolyData plot() 
    function to  visualize the mesh model in 3D. To save it to a VTK 
    file, use the PyVista save() function. Laplacian smoothing can also
    be applied the surface mesh before it is returned. The inputs
    for this function are based on the outputs of 
    convert_voxels_to_surface() defined above.

    ---- INPUT ARGUMENTS ----
    [[verts_in]]: A 2D array that contains the coordinates of the mesh 
        vertices. The array should have 3 columns, such that each row
        will provide the [X, Y, Z] coordinates of that vertex. Assuming
        there are V unique mesh vertices, then the size is (V, 3).

    [[faces_in]]: A 2D array that defines the connectivity of the 
        triangular faces. Assuming there are F faces, then each face 
        will require three nodes to define it. These indices of the
        desired nodes from verts_in are used to define this 
        connectivity. The size of faces_in should be (F, 3).

    [[values_in]]: A 1D array corresponding to the values that should
        be assigned to each vertex. These will be saved as a point_array
        in the pyvista.PolyData object, called "values".

    smth_iter: A scalar integer that defines the number of iterations
        of Laplacian smoothing to be applied. A value of 0 will prevent
        any smoothing from occurring. 

    ---- RETURNED ----
    surf: A PyVista PolyData object that represents the triangular mesh.
        For more information, go to:

        https://docs.pyvista.org/core/points.html?highlight=polydata#
        pyvista.PolyData

    ---- SIDE EFFECTS ---- 
    Function input arguments should not be altered. However, a new view
    is created for the vertices and faces. So, this cannot be guaranteed. 
    Nothing is written to the hard drive.
    """

    verts = verts_in
    faces = faces_in
    values = values_in
    smth_iter = np.round(smth_iter)

    num_faces = faces.shape[0]
    vec_of_3s = (np.ones((num_faces,1))*3)
    faces2 = np.hstack( np.column_stack((vec_of_3s,faces)) ).astype(np.uint32)
    surf = pv.PolyData(verts, faces2)

    if smth_iter > 0:
        surf = surf.smooth(n_iter=smth_iter, inplace=False)

    surf.point_data["values"] = values
    surf.compute_normals(inplace=True)

    return surf


def make_vtk_unstruct_grid_slow(img_arr_in):
    """
    This function converts an image sequence (i.e., a 3D array that
    contains 2D images) into an unstructured VTK voxel model. Although
    this function creates the voxel model in an efficient manner
    concerning RAM, it is super slow thanks to some big Python 
    for-loops. Best suited for models that are less than 128 by 128 by
    128 pixels in total size. For a faster implementation based on a
    custom C-extension, see make_vtk_unstruct_grid() below. Note, this
    function will only convert pixels with intensity values greater than
    or equal to one into voxels -- black pixels are skipped.

     ---- INPUT ARGUMENTS ---- 
    [[[img_arr_in]]]: A 3D Numpy array representing the image sequence
        It is important that this is a Numpy array and not a Python list
        of Numpy matrices. The shape of img_arr_in is expected to be as
        (num_images, num_pixel_rows, num_pixel_cols). Note, when
        converting this to spatial coordinates, num_images is the
        Z-component, num_pixel_rows is the Y-component, and
        num_pixel_cols is the X-component. It is expected that the
        images are single-channel, and the data should be of type uint8.
        Specifically, intensities should range from 0 to 255. If it is
        segmented (i.e., binarized), then only values of 0 and 255 
        should be present (NOT 0 and 1).

    ---- RETURNED ----
    unstruct_grid: A pyvista.UnstructuredGrid object is returned. 
        PyVista is a Python library that acts as a high-level wrapper
        for VTK - the visualization toolkit. PyVista's UnstructuredGrid
        object is an extension of VTK's vtkUnstructuredGrid class. For
        more information, see 
        https://docs.pyvista.org/core/point-grids.html

    ---- SIDE EFFECTS ----
    Function input arguments should not be altered. However, a new view
    is created for img_arr_in, not a deep copy. So, this cannot be 
    guaranteed. Nothing is written to the hard drive.
    """

    img_arr = img_arr_in # Do not transpose here!
    img_arr_shape = img_arr.shape
    n_imgs = img_arr_shape[0]
    n_rows = img_arr_shape[1]
    n_cols = img_arr_shape[2]

    # The number of voxels that will be created. Initialize the Numpy
    # connectivity array that will be used to create the VTK model
    num_cells = np.count_nonzero(img_arr >= 1)
    cells = np.zeros((num_cells,9), dtype=np.uint32)
    vals = np.ones(num_cells, dtype=np.uint8)*255
    vert_ids = np.ones((n_imgs+1, n_rows+1, n_cols+1), dtype=np.uint32)\
        *4294967295 # Largest possible integer in a uint32 data type

    node_arr = []
    nd_k = 0 # Node IDs start at zero
    cell_k = 0 # Cell IDs start at zero
    for ii in range(0, n_imgs):
        for rr in range(0, n_rows):
            for cc in range(0, n_cols):

                cur_int = img_arr[ii,rr,cc]
                if cur_int < 1:
                    continue # Skip black voxels

                xc = cc # Converting to XYZ coordinates here
                yc = rr
                zc = ii

                # Construct the node array spatial coordinates 
                nd1 = np.array([xc-0.5, yc-0.5, zc-0.5]).astype(np.float32)
                nd2 = np.array([xc+0.5, yc-0.5, zc-0.5]).astype(np.float32)
                nd3 = np.array([xc+0.5, yc+0.5, zc-0.5]).astype(np.float32)
                nd4 = np.array([xc-0.5, yc+0.5, zc-0.5]).astype(np.float32)
                nd5 = np.array([xc-0.5, yc-0.5, zc+0.5]).astype(np.float32)
                nd6 = np.array([xc+0.5, yc-0.5, zc+0.5]).astype(np.float32)
                nd7 = np.array([xc+0.5, yc+0.5, zc+0.5]).astype(np.float32)
                nd8 = np.array([xc-0.5, yc+0.5, zc+0.5]).astype(np.float32)

                if vert_ids[ii, rr, cc] == 4294967295: # Node 1 not being used
                    vert_ids[ii, rr, cc] = nd_k
                    nd1_id = nd_k
                    node_arr.append(nd1)
                    nd_k += 1
                else:
                    nd1_id = vert_ids[ii, rr, cc]

                if vert_ids[ii, rr, cc+1] == 4294967295: # Node 2 not being used
                    vert_ids[ii, rr, cc+1] = nd_k
                    nd2_id = nd_k
                    node_arr.append(nd2)
                    nd_k += 1
                else:
                    nd2_id = vert_ids[ii, rr, cc+1]

                if vert_ids[ii, rr+1, cc+1] == 4294967295: # Node 3 not being used
                    vert_ids[ii, rr+1, cc+1] = nd_k
                    nd3_id = nd_k
                    node_arr.append(nd3)
                    nd_k += 1
                else:
                    nd3_id = vert_ids[ii, rr+1, cc+1]

                if vert_ids[ii, rr+1, cc] == 4294967295: # Node 4 not being used
                    vert_ids[ii, rr+1, cc] = nd_k
                    nd4_id = nd_k
                    node_arr.append(nd4)
                    nd_k += 1
                else:
                    nd4_id = vert_ids[ii, rr+1, cc]

                if vert_ids[ii+1, rr, cc] == 4294967295: # Node 5 not being used
                    vert_ids[ii+1, rr, cc] = nd_k
                    nd5_id = nd_k
                    node_arr.append(nd5)
                    nd_k += 1
                else:
                    nd5_id = vert_ids[ii+1, rr, cc]

                if vert_ids[ii+1, rr, cc+1] == 4294967295: # Node 6 not being used
                    vert_ids[ii+1, rr, cc+1] = nd_k
                    nd6_id = nd_k
                    node_arr.append(nd6)
                    nd_k += 1
                else:
                    nd6_id = vert_ids[ii+1, rr, cc+1]

                if vert_ids[ii+1, rr+1, cc+1] == 4294967295: # Node 7 not being used
                    vert_ids[ii+1, rr+1, cc+1] = nd_k
                    nd7_id = nd_k
                    node_arr.append(nd7)
                    nd_k += 1
                else:
                    nd7_id = vert_ids[ii+1, rr+1, cc+1]

                if vert_ids[ii+1, rr+1, cc] == 4294967295: # Node 8 not being used
                    vert_ids[ii+1, rr+1, cc] = nd_k
                    nd8_id = nd_k
                    node_arr.append(nd8)
                    nd_k += 1
                else:
                    nd8_id = vert_ids[ii+1, rr+1, cc]

                cells[cell_k,:] = np.array([8, nd1_id, nd2_id, nd3_id, nd4_id,\
                    nd5_id, nd6_id, nd7_id, nd8_id])

                vals[cell_k] = cur_int

                cell_k += 1

    del vert_ids # No longer need this 

    # Convert this back to a 2D numpy array. No need to keep the values. 
    # The indices themselves for each row are the values (i.e., node IDs).
    #coords = np.array(list(coords.keys()))
    node_arr = np.array(node_arr)

    # Different ways to initialize the grid if using VTK version 9.0 or newer
    vtk_ver_junk_str = vtk.vtkVersion.GetVTKSourceVersion()
    vtk_ver_str = ''.join(c for c in vtk_ver_junk_str if c.isdigit())
    vtk_ver_num = int(vtk_ver_str[0])

    if vtk_ver_num >= 9: 
        # I don't have Version 9 of VTK, so this is untested
        # Using PyVista example for this code sample:
        #
        # https://docs.pyvista.org/examples/00-load/create-unstructured-
        # surface.html#sphx-glr-examples-00-load-create-unstructured-surface-py

        unstruct_grid = pv.UnstructuredGrid({vtk.VTK_HEXAHEDRON: cells[:, 1:]}, 
            node_arr)

    else:
        # Flattens this into a 1D-array (using C-like row-major order)
        cells = cells.ravel()

        # Offset/index index to go to each cell for the flattened array
        off_end = num_cells*9
        offset = np.arange(0, off_end, 9, dtype=np.uint32)

        # Each cell is a VTK_HEXAHEDRON
        cell_types = np.empty(num_cells, dtype=np.uint8)
        cell_types[:] = vtk.VTK_HEXAHEDRON

        unstruct_grid = pv.UnstructuredGrid(offset, cells, cell_types, node_arr)

    # Assign the intensity values to the cells
    unstruct_grid.cell_data["values"] = vals

    return unstruct_grid


def make_vtk_unstruct_grid(img_arr_in, scale_spacing=1.0, 
    all_voxels=False, quiet_in=False):
    """
    This function converts an image sequence into an unstructured VTK
    voxel model. This function depends on a C-extension binary that
    was created using Cython. This function is the super fast version
    of make_vtk_unstruct_grid_slow(), and in general, is preferred.
    However, the binary file, called im3_processing.cp37-win_amd64.pyd
    was compiled for Python 3.7 for Windows x64. If you are using a 
    different operating system or version of Python, then the .pyx
    file will need to be re-compiled for a new target. Contact Newell
    Moser for assistance if this is the case. Otherwise, you will need
    to re-compile it or use make_vtk_unstruct_grid_slow(). By default,
    this function will only convert pixels with intensity values greater
    than or equal to one -- black pixels are skipped. However, if RAM
    is not an issue, then set all_voxels equal to True to convert all
    of the pixels. Use PyVista UniformGrid plot() function to 
    visualize the uniform grid model in 3D. To save it to a VTK file,
    use the PyVista save() function. Note, the intensity values will
    be saved in the uniform grid under the name "values".

     ---- INPUT ARGUMENTS ---- 
    [[[img_arr_in]]]: A 3D Numpy array representing the image sequence
        It is important that this is a Numpy array and not a Python list
        of Numpy matrices. The shape of img_arr_in is expected to be as
        (num_images, num_pixel_rows, num_pixel_cols). Note, when
        converting this to spatial coordinates, num_images is the
        Z-component, num_pixel_rows is the Y-component, and
        num_pixel_cols is the X-component. It is expected that the
        images are single-channel, and the data should be of type uint8.
        Specifically, intensities should range from 0 to 255. If it is
        segmented (i.e., binarized), then only values of 0 and 255 
        should be present (NOT 0 and 1).

    scale_spacing: A scalar float that represents the distance between
        the voxels before they are converted to a surface. If this is
        an X-ray CT image sequence, this represents the voxel size.

    all_voxels: By default, this is set to False, which means only non-
        black pixels will be converted to voxels. Specifically, pixels
        with intensites greater than or equal to one. Consequently, 
        this function will return a pyvista.UnstructuredGrid object. Set
        this to True in order to return all of the pixels as voxels, 
        in which case, a pyvista.StructuredGrid object is returned.

    quiet_in: An optional boolean that is by default set to False. Set 
        to True to prevent outputting any messages. 

    ---- RETURNED ----
    unstruct_grid: If all_voxels is False, a pyvista.UnstructuredGrid 
        object is returned. PyVista is a Python library that acts as a
        high-level wrapper for VTK - the visualization toolkit. 
        PyVista's UnstructuredGrid object is an extension of VTK's 
        vtkUnstructuredGrid class. If If all_voxels is True, then a 
        pyvista.StructuredGrid object is returned. PyVista's 
        StructuredGrid object is an extension of VTK's 
        vtk.vtkStructuredGrid object. For more information, see 
        https://docs.pyvista.org/core/point-grids.html

    ---- SIDE EFFECTS ----
    Function input arguments should not be altered. However, a new view
    is created for img_arr_in, not a deep copy. So, this cannot be 
    guaranteed. Nothing is written to the hard drive.
    """

    # Do NOT transpose here. My C-extension expects the image array
    # in image coordinates, and will return it correctly in XYZ.
    img_arr = img_arr_in 

    n_imgs = img_arr_in.shape[0]
    n_rows = img_arr_in.shape[1]
    n_cols = img_arr_in.shape[2]

    quiet = quiet_in

    if all_voxels: # Without downsampling, this WILL kill your RAM
        if not quiet:
            print("\nAttempting to construct the structured VTK file...")

        unstruct_grid = make_vtk_uniform_grid(img_arr)
        unstruct_grid = unstruct_grid.cast_to_structured_grid()
        
        if not quiet:
            print("  Success!")

    else:
        # Call my Cython C-extension. Set the second argument, "debug_flag",
        # to 1 (instead of 0) if you want the expected memory allocations to 
        # also be written to the terminal standard output.
        if not quiet:
            print("\nAllocating memory for the VTK voxel data structure...")
        
        nd_coords, cell_conn, cell_vals = im3.get_voxel_info(img_arr, 0)
        num_cells = cell_conn.shape[0]

        nd_coords = nd_coords*scale_spacing

        # Different ways to initialize the grid if using VTK version 9.0 or newer
        vtk_ver_junk_str = vtk.vtkVersion.GetVTKSourceVersion()
        vtk_ver_str = ''.join(c for c in vtk_ver_junk_str if c.isdigit())
        vtk_ver_num = int(vtk_ver_str[0])

        if not quiet:
            print("\nAttempting to construct the unstructured VTK file...")
        
        if vtk_ver_num >= 9: 
            # I don't have Version 9 of VTK, so this is untested
            # Using a PyVista example for this code sample:
            #
            # https://docs.pyvista.org/examples/00-load/create-unstructured-
            # surface.html#sphx-glr-examples-00-load-create-unstructured-surface-py

            unstruct_grid = pv.UnstructuredGrid(
                {vtk.VTK_HEXAHEDRON: cell_conn[:, 1:]}, nd_coords)

        else:
            # Flattens this into a 1D-array (using C-like row-major order)
            cell_conn = cell_conn.ravel()

            # Offset/index index to go to each cell for the flattened array
            off_end = num_cells*9
            offset = np.arange(0, off_end, 9, dtype=np.uint32)

            # Each cell is a VTK_HEXAHEDRON
            cell_types = np.empty(num_cells, dtype=np.uint8)
            cell_types[:] = vtk.VTK_HEXAHEDRON

            unstruct_grid = pv.UnstructuredGrid(offset, cell_conn, 
                cell_types, nd_coords)

        # Assign the intensity values to the cell_conn
        unstruct_grid.cell_data["values"] = cell_vals

    if not quiet:
        print("  Success!")

    return unstruct_grid