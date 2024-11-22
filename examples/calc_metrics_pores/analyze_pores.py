"""
Written by Newell Moser
NRC Postdoctoral Fellow
National Institute of Standards and Technology
November 12, 2024

This script calculates various metrics of pores, like equivalent 
spherical diameter. This is particularly relevant for metal-based
additive manufacturing. 

The input to this example is a binarized (or segmented) image stack 
where white pixels correspond to metal/solid material. Examples on
segmenting image stacks can be found in "../interactive_filtering/",
such as "segment_scikit_single_image.py" and 
"interactive_scikit_processing_single_image.py". This example script
 uses synthetic data found in the resources folder,
"../resources/porous_metal/". If no images are in this folder, then
please generate the sample data before running this example, which can 
be easily done by executing "../resources/generate_sample_data.py".

In addition to printing messages to the console, various files will also 
be saved to the hard drive during this example. The identified pores
will be saved in a separate, image stack in the subdirectory,
"./labeled_pores/". In this image stack, fully enclosed pores will be
denoted by gray pixels. The grayscale intensity of any given pore will
be equal to that pore's ID number. For example, the pore labeled as "22"
will be represented by pixels with grayscale intensities equal to 22.

Moreover, 3D models will be created of the pores and the solid metal
surface. These models will be saved as .vtk files, which can be opened
by ParaView. In the .vtk voxel model of the pores, various metrics will
be saved as field variables, including the label (i.e., number) of the
pore and its equivalent spherical diameter.

Finally, a .csv file will be saved which contains a summary of all the
metrics for each pore. The number of rows in the .csv file will be equal
to the number of pores, and there will be 24 columns, each representing 
a different quantity or metric related to a given pore. Additional
information about each column in the .csv file is described below based
on the header strings:

    1) “Void ID”: This is the identification number, or label, of the 
    pore/void.

    2) “Voxel Size [um/pixel]”: The voxel size (or pixel size) used in
    the XCT measurement. It relates microns to pixels, and it does not
    change for any of the data sets.

    3) “Num. Voxels in Void”: The number of voxels in the pore. 

    4) “Volume of Void [um^3]”: The volume of the pore in μm^3, which is
    simply the number of the voxels multiplied by the voxel-size-cubed.

    5) “Equiv. Spher. Diameter [um]”: The equivalent spherical diameter
    of the pore in μm. This is calculated using the volume of the pore
    and then using the volume of a sphere to get an effective diameter,
    d_eff = 2 (3V⁄4π)^(1/3).

    6) “Approx. Sphericity”: The sphericity of the pore relates the 
    volume of the pore to its surface area, Ψ = (π^(1⁄3) (6V)^(2⁄3))⁄A. 
    To improve the accuracy of this calculation, the pore is first 
    converted to a surface-based mesh using the marching cubes 
    algorithm. Note, this field value will read “NaN” if the pore 
    contains fewer than 64 voxels. 

    7) “Centroid Img Num. (0-based indexing)”: The coordinates of the 
    centroid can be described using indices representing the image 
    number, row number, and column number. This specific data field 
    represents the image index of the centroid. Note, 0-based indexing
    is used here, implying that the first image corresponds to zero, not
    one. 

    8) “Centroid Row Num. (0-based indexing)”: The row index of the 
    centroid of the pore, using 0-based indexing. For a given image, the
    first row starts at the top of the image. Also, see the notes above
    for bullet 7.

    9) “Centroid Col Num. (0-based indexing)”: The column index of the
    centroid of the pore, using 0-based indexing. For a given image, the
    first column starts on the left-side of the image. Also, see the
    notes above for bullet 7.

    10. “Centroid VTK X [um]”: When visualizing the 3D voids in ParaView
    using the VTK-voxel files, there is a need to convert the image
    indices into XYZ-coordinates. For complete transparency, the
    XYZ-coordinates of the centroids of the pores in VTK models
    are given in microns. This column specifically provides the
    X-coordinate (in microns) of the given centroid in the VTK model.

    11. “Centroid VTK Y [um]”: The Y-coordinate (in microns) of the
    given centroid in the VTK model. In order to maintain a right-handed
    coordinate system, the Y-coordinate has been flipped about the 
    origin relative to the image-index coordinates. Also, see notes 
    above for bullet 10.

    12. “Centroid VTK Z [um]”: The Z-coordinate (in microns) of the
    given centroid in the VTK model. Also, see notes above for bullet
    10.

    13) “BBox Center Img Num. (0-based indexing)”: A rotated bounding
    box was fitted to pores containing at least 64 voxels. The three
    lengths are denoted based on their lengths, L≥W≥T. Specifically, L
    is defined as the longest distance between (the centers of) any two
    voxels in a given pore. W is the longest distance between any to
    voxels while remaining orthogonal to L. T is the longest distance
    between any to voxels while remaining orthogonal to L and W. This
    specific data field represents the image index of the center of the
    bounding box. Note, 0-based indexing is used here, implying that the
    first image corresponds to zero, not one. Will be “NaN” if the pore
    contains fewer than 64 voxels.

    14) “BBox Center Row Num. (0-based indexing)”: The row index of the
    center of the bounding box, using 0-based indexing. For a given
    image, the first row starts at the top of the image. Also, see the
    notes above for bullet 10. Will be “NaN” if the pore contains fewer
    than 64 voxels.

    15) “BBox Center Col Num. (0-based indexing)”: The column index of
    the center of the bounding box, using 0-based indexing. For a given
    image, the first column starts on the left-side of the image. Also,
    see the notes above for bullet 7. Will be “NaN” if the pore
    contains fewer than 64 voxels.

    16) “Rotated BBox (L)ength [um]”: The length, L, of the bounding box
    in μm where L ≥ W ≥ T. Also, see the notes above for bullet 10. Will
    be “NaN” if the pore contains fewer than 64 voxels.

    17) “Rotated BBox (W)idth [um]”: The length, W, of the bounding box
    in μm where L ≥ W ≥ T. Also, see the notes above for bullet 10. Will
    be “NaN” if the pore contains fewer than 64 voxels.

    18) “Rotated BBox (T)hickness [um]”: The length, T, of the bounding
    box in μm where L ≥ W≥ T. Also, see the notes above for bullet 10.
    Will be “NaN” if the pore contains fewer than 64 voxels.

    19) “L_Vec. Img Component”: The orientations of the three lengths of
    the bounding box can be defined via directional unit vectors. This
    was done using image indices. In other words, each vector will have
    an image index component, row index component, and a column index
    component. This specific data field represents image index
    component of the unit vector collinear with L. Will be “NaN” if the
    pore contains fewer than 64 voxels.

    20) “L_Vec. Row Component”: The row component of the unit vector
    collinear with L. Also, see the notes above for bullet 16. Will
    be “NaN” if the pore contains fewer than 64 voxels.

    21) “L_Vec. Col Component”: The column component of the unit vector
    collinear with L. Also, see the notes above for bullet 16. Will
    be “NaN” if the pore contains fewer than 64 voxels.

    22) “W_Vec. Img Component”: The image component of the unit vector
    collinear with W. Also, see the notes above for bullet 16. Will
    be “NaN” if the pore contains fewer than 64 voxels. 

    23) “W_Vec. Row Component”: The row component of the unit vector
    collinear with W. Also, see the notes above for bullet 16. Will
    be “NaN” if the pore contains fewer than 64 voxels.

    24) “W_Vec. Col Component”: The column component of the unit vector
    collinear with W. Also, see the notes above for bullet 16. Will
    be “NaN” if the pore contains fewer than 64 voxels.

    25) “T_Vec. Img Component”: The image component of the unit vector
    collinear with T. Also, see the notes above for bullet 16. Will
    be “NaN” if the pore contains fewer than 64 voxels.

    26) “T_Vec. Row Component”: The row component of the unit vector
    collinear with T. Also, see the notes above for bullet 16. Will
    be “NaN” if the pore contains fewer than 64 voxels.

    27) “T_Vec. Col Component”: The column component of the unit vector
    collinear with T. Also, see the notes above for bullet 16. Will
    be “NaN” if the pore contains fewer than 64 voxels.
"""

# Import external dependencies
import os, csv
import numpy as np
from skimage import measure as meas
from skimage import segmentation as sseg
from skimage import morphology as morph
from skimage.util import img_as_float, img_as_ubyte
from skimage.util import invert as ski_invert

# Import local modules
import imppy3d.import_export as imex
import imppy3d.volume_image_processing as vol
import imppy3d.bounding_box as box
import imppy3d.vtk_api as vapi


# A function that converts a 2D python list of values to strings.
# The level of precision can be controlled using str_fmt_in.
def convert_2d_list_to_str(list_in, str_fmt_in):

    list_out = []
    for cur_row in list_in:
        cur_row_str = []

        for m, cur_val in enumerate(cur_row):
            if cur_val == -999:
                cur_row_str.append('NaN')
            elif (str_fmt_in[m].upper()) == 'E':
                cur_row_str.append(f"{cur_val:.6e}")
            elif (str_fmt_in[m].upper()) == 'F':
                cur_row_str.append(f"{cur_val:.4f}")
            elif (str_fmt_in[m].upper()) == 'I':
                cur_row_str.append(f"{cur_val:.0f}")
            else:
                cur_row_str.append(str(cur_val))

        list_out.append(cur_row_str)

    return list_out


# A function to save a 2D python list to the hard drive as a .csv file.
def save_2d_csv_file(list_in, str_fmt_in, path_in, header_in):

    list_in_str = convert_2d_list_to_str(list_in, str_fmt_in)

    with open(path_in, 'w', newline='') as file_obj:
        csv_writer = csv.writer(file_obj, delimiter=',')

        csv_writer.writerow(header_in)

        for cur_row in list_in_str:
             csv_writer.writerow(cur_row)


# A helper function for using the BBox class in IMPPY3D.
def bbox_search(imgs_in, search_flag_in):

     # ---------- FUNCTION INPUTS ----------
    #
    # imgs_in: A Numpy image stack (i.e., 3D array) of data type
    # np.uint8. The images should be binarized such that there are
    # only black pixels (with zero intensity) and white pixels (with
    # 255 intensity). This function will fit a rotated bounding box
    # around all of the white pixels.
    #
    # search_flag_in: An integer that defines what type of algorithm
    # to use in calculating the rotated bounded box. Can be either 0,
    # 1, 2, 3, or 4. See below.
    # 
    # NOTE: In general, the rotated, minimum-volume bounding box of a
    # particle does not do a good job at capturing the general
    # orientation of the particle itself. So, the most accurate
    # algorithms, such as search_flag_in = 0, will not necessarily give
    # the desired results. Our recommendation is to use 
    # search_flag_in = 3 or 4.
    # 
    #  0: Use an exhaustive search. Although not necessarily the
    #  fastest, this method should find the best bounding box within 1
    #  degree of any arbitrary rotation. This method is the default
    #  search algorithm.
    #
    #  1: Perform a singular-value decomposition calculation on the data
    #  set and use the eigenvector corresponding to the minimum
    #  variation (i.e., smallest eigenvalue) as one of the directions
    #  of the bounding box. The other two directions are found by
    #  projecting the points onto a plane normal to this minimum
    #  principal direction, and then solving the 2D bounding box
    #  problem using an exhaustive search. In practice, this method
    #  usually is slightly more accurate than using the maximum
    #  variation (see search=2 below) while being just as
    #  computationally efficient. Performing the convex hull usually
    #  improves the accuracy for this search.
    #
    #  2: Perform a singular-value decomposition calculation on the data
    #  set and use the eigenvector corresponding to the maximum
    #  variation (i.e., largest eigenvalue) as one of the directions of
    #  the bounding box. The other two directions are found by
    #  projecting the points onto a plane normal to this maximum
    #  principal direction, and then solving the 2D bounding box
    #  problem using an exhaustive search. Performing the convex hull
    #  usually improves the accuracy for this search algorithm.
    #
    #  3: Perform a singular-value decomposition calculation on the data
    #  set and use all three eigenvectors as directions of the bounding
    #  box. Although this is the fastest method for large data sets,
    #  this is also the least reliable method in terms of accuracy.
    #  Performing the convex hull usually improves the accuracy for
    #  this search algorithm.
    #
    #  4: Find the longest distance between two points and define this
    #  direction to be major z-axis of the bounding box. The other two
    #  directions are found by projecting the points onto a plane
    #  normal to this maximum principal direction, and then solving the
    #  2D bounding box problem using an exhaustive search.
    #
    # ---------- FUNCTION RETURNS ----------
    #
    # bbox_fit: Returns the bounding box object (BBox class)
    #
    # b_irc_props: A Numpy array that contains the lengths of the  
    # bounding box, and the vector-directions of each of these lengths.  
    # They have been returned in image coordinates as such:
    # [[max_length, img_dir, row_dir, col_dir],
    #  [middle_length, img_dir, row_dir, col_dir],
    #  [min_length, img_dir, row_dir, col_dir]]
    #
    # bbox_center: The center of the bounding box given in image
    # coordinates (as a Numpy array) as such:
    # [img_position, row_position, col_position]


    # Find the outermost white pixels along the feature boundary
    img_bound = sseg.find_boundaries(imgs_in, connectivity=1,
        mode='inner', background=0)

    img_bound = img_as_ubyte(img_bound) # Convert from bool to uint8

    # Create a 2D list of the indices to the boundary pixels.
    # [[row_1, col_1], [row_2, col_2], ..., [row_n, col_n]]
    bound_coords = np.argwhere(img_bound)
    bound_coords_xyz = vol.intrinsic_to_spatial(bound_coords)

    bbox_fit = box.min_bounding_box(bound_coords_xyz, search=search_flag_in)

    b_x_props = np.hstack((np.array(bbox_fit.x_len), bbox_fit.calc_x_vec()))
    b_y_props = np.hstack((np.array(bbox_fit.y_len), bbox_fit.calc_y_vec()))
    b_z_props = np.hstack((np.array(bbox_fit.z_len), bbox_fit.calc_z_vec()))

    b_irc_props = np.row_stack((b_x_props, b_y_props, b_z_props))
    b_irc_props[:,1:4] = vol.spatial_to_intrinsic(b_irc_props[:,1:4])

    # Sort array in ascending order based on the first column
    ind = np.argsort(b_irc_props[:,0])
    b_irc_props = b_irc_props[ind]

    bbox_center = vol.spatial_to_intrinsic(bbox_fit.center)

    return (bbox_fit, b_irc_props, bbox_center)


if __name__ == '__main__':

    # -------- USER INPUTS --------

    # Variables needed to import the binarized image stack
    imgs_dir_in_path = "../resources/porous_metal/"
    imgs_name_in_substr = "metal"
    imgs_keep = (256,) # Import all images
    rev_img_stack = False

    # Variables need to save the image stack of labeled pores. See
    # comments at the top for more details.
    imgs_pores_dir_out_path = "./labeled_pores/"
    imgs_name_out_substr = "gray_pores.tif"

    # The pixel size, which is arbitrarily chosen for this example.
    voxel_width_um = 5.0 # um/pixel

    # Minimum number of voxels required to calculate certain metrics
    # on the pores. For example, calculating shape-related metrics, 
    # such as sphericity, requires a certain degree of resolution.
    min_feat_size = 4*4*4

    # The path of the .csv file that will be written to which contains
    # all of the calculated pore metrics.
    csv_pore_path_out = "./pore_props.csv"

    # The path of the .vtk file containing all of the pores as voxel
    # models. This file also contains field names for various metrics
    # such as number of voxels and equivalent spherical diameter.
    vtk_pore_path_out = "./pores_voxels.vtk"

    # The of the .vtk file containing a mesh representing the boundaries
    # of the solid (or metallic) material.
    vtk_solid_path_out = "./solid_mesh.vtk"


    # -------- DEFINITIONS OF CONSTANTS --------

    # Create the directory if it is not there already
    os.makedirs(imgs_pores_dir_out_path, exist_ok=True)

    # The header string used in the .csv file of pore metrics. See
    # comments at the top.
    void_props_header_str = [
        'Void ID', #1
        'Voxel Size [um/pixel]', #2
        'Num. Voxels in Void', #3
        'Volume of Void [um^3]', #4
        'Equiv. Spher. Diameter [um]', #5
        'Approx. Sphericity', #6
        'Centroid Img Num. (0-based indexing)', #7
        'Centroid Row Num. (0-based indexing)', #8
        'Centroid Col Num. (0-based indexing)', #9
        'Centroid VTK X [um]', #10
        'Centroid VTK Y [um]', #11
        'Centroid VTK Z [um]', #12
        'BBox Center Img Num. (0-based indexing)', #13
        'BBox Center Row Num. (0-based indexing)', #14
        'BBox Center Col Num. (0-based indexing)', #15
        'Rotated BBox (L)ength [um]', #16
        'Rotated BBox (W)idth [um]', #17
        'Rotated BBox (T)hickness [um]', #18
        'L_Vec. Img Component', #19
        'L_Vec. Row Component', #20
        'L_Vec. Col Component', #21
        'W_Vec. Img Component', #22
        'W_Vec. Row Component', #23
        'W_Vec. Col Component', #24
        'T_Vec. Img Component', #25
        'T_Vec. Row Component', #26
        'T_Vec. Col Component', #27
        ]

    # Format variables to control the precision of the saved quantities
    # in the .csv file.
    void_props_fmt_str = [
        'I', #1
        'F', #2
        'I', #3
        'E', #4
        'F', #5
        'F', #6
        'F', #7
        'F', #8
        'F', #9
        'F', #10
        'F', #11
        'F', #12
        'F', #13
        'F', #14
        'F', #15
        'F', #16
        'F', #17
        'F', #18
        'F', #19
        'F', #20
        'F', #21
        'F', #22
        'F', #23
        'F', #24
        'F', #25
        'F', #26
        'F', #27
        ]

    # Some constants related to voxel size, to be used later.
    voxel_width_mm = voxel_width_um/1000.0 # mm/pixel
    voxel_width_um2 = voxel_width_um**2
    voxel_width_um3 = voxel_width_um**3


    # -------- IMPORT BINARY IMAGE SEQUENCE --------

    print(f"\nImporting binary images...")
    imgs_solid_bin, imgs_names = imex.load_image_seq(imgs_dir_in_path, 
        file_name_in=imgs_name_in_substr, indices_in=imgs_keep,
        flipz=rev_img_stack)


    # -------- INVERT THE IMAGES AND ISOLATE PORES --------

    print(f"\nInverting the image stack and applying flood fill...")
    # Pores will now be white, including the exterior air
    imgs_pore_bin = ski_invert(imgs_solid_bin)
    imgs_pore_bin = img_as_ubyte(imgs_pore_bin)

    # Flood fill the exterior. All that remains should be fully
    # enclosed pores.
    seed_coord = (0,0,0)
    imgs_pore_bin = sseg.flood_fill(imgs_pore_bin, seed_point=seed_coord,
        new_value=0, connectivity=1)

    # (OPTIONAL) Remove pores that are too small to characterize
    resolution_limit = 2**3 # Minimum number of voxels
    
    print(f"\nRemoving particles with fewer than {resolution_limit} voxels...")
    img_arr_label = meas.label(imgs_pore_bin, connectivity=2)

    # Converting back to a 8-bit image with pores being white (255)
    imgs_pore_bin = morph.remove_small_objects(img_arr_label, 
        min_size=resolution_limit, connectivity=2)

    imgs_pore_bin[np.nonzero(imgs_pore_bin)] = 255
    imgs_pore_bin = img_as_ubyte(imgs_pore_bin)


    # -------- LABEL THE PORES --------

    print(f"\nLabeling connected regions...")
    img_arr_label = meas.label(imgs_pore_bin, connectivity=1)
    feat_props = meas.regionprops(img_arr_label)
    n_feats = len(feat_props) # Number of voids

    void_props = []
    vtk_list_pores = []

    if n_feats < 255:
        imgs_labeled_out = np.zeros(imgs_pore_bin.shape, dtype=np.uint8)
    elif n_feats < 65535:
        imgs_labeled_out = np.zeros(imgs_pore_bin.shape, dtype=np.uint16)
    else:
        imgs_labeled_out = np.zeros(imgs_pore_bin.shape, dtype=np.uint32)


    # -------- LOOP THROUGH EACH PORE AND CALCULATE METRICS --------
    
    for void_index, cur_feat in enumerate(feat_props):
        
        # Unique pore identifier that starts at one
        void_id = void_index + 1

        # Grab an image sub-stack of just the pore feature 
        cur_img = img_as_ubyte(cur_feat.image)

        # The global (img, row, col) coordinates of the white pixels
        cur_coords = cur_feat.coords

        # Current pore sub-stack that is also filled in
        cur_img_fill = img_as_ubyte(cur_feat.filled_image)

        # Translation vector in (img, row, col) pixel coordinates
        trans_vec = np.array([cur_feat.bbox[0], cur_feat.bbox[1], cur_feat.bbox[2]])
        trans_vec_xyz = voxel_width_um*(vol.intrinsic_to_spatial(trans_vec))

        # Centroid of the pore (in original coordinate system)
        pore_cent = cur_feat.centroid # (img_index, row_index, column_index)
        pore_cent = np.array([pore_cent[0], pore_cent[1], pore_cent[2]])

        # Pore volume based on voxel model, converted to microns-cubed
        num_void_voxels = np.count_nonzero(cur_img)
        void_vol = num_void_voxels*voxel_width_um3 # [um^3]

        # Equivalent spherical diameter in [um]
        eqv_diam = 2.0*np.cbrt(0.23873241463*void_vol)

        # Create voxel VTK model of the void
        voxel_void = vapi.make_vtk_unstruct_grid(cur_img_fill, 
            scale_spacing=voxel_width_um, quiet_in=True)

        voxel_void.translate(tuple(trans_vec_xyz), inplace=True)
        voxel_void.flip_y(point=[0.0, 0.0, 0.0], inplace=True)

        # Calculate the center of each void in the VTK model
        cell_centers_obj = voxel_void.cell_centers() # PolyData object
        cell_centers_arr = cell_centers_obj.points # 2D Numpy array
        void_vtk_cent = np.mean(cell_centers_arr, axis=0) # Centroid [um]

        pnt_cent_vec_arr = np.ones((voxel_void.n_points, 3), dtype=np.float32)
        pnt_cent_vec_arr[:,0] = void_vtk_cent[0]
        pnt_cent_vec_arr[:,1] = void_vtk_cent[1]
        pnt_cent_vec_arr[:,2] = void_vtk_cent[2]

        cell_cent_vec_arr = np.ones((voxel_void.n_cells, 3), dtype=np.float32)
        cell_cent_vec_arr[:,0] = void_vtk_cent[0]
        cell_cent_vec_arr[:,1] = void_vtk_cent[1]
        cell_cent_vec_arr[:,2] = void_vtk_cent[2]

        # Add fields to the VTK model
        voxel_void.point_data['Pore ID'] = void_id*np.ones(voxel_void.n_points,
            dtype=np.uint16)

        voxel_void.point_data['Num Voxels'] = num_void_voxels*np.ones(
            voxel_void.n_points, dtype=np.uint32)

        voxel_void.point_data['Eqv Sph Diam [um]'] = eqv_diam*np.ones(
            voxel_void.n_points, dtype=np.float32)

        voxel_void.point_data['Centroid [um]'] = pnt_cent_vec_arr

        voxel_void.cell_data['Pore ID'] = void_id*np.ones(voxel_void.n_cells,
            dtype=np.uint16)

        voxel_void.cell_data['Num Voxels'] = num_void_voxels*np.ones(
            voxel_void.n_cells, dtype=np.uint32)

        voxel_void.cell_data['Eqv Sph Diam [um]'] = eqv_diam*np.ones(
            voxel_void.n_cells, dtype=np.float32)

        voxel_void.cell_data['Centroid [um]'] = cell_cent_vec_arr

        # Save the labeled pore in a new 16-bit image stack
        ii_arr = tuple(cur_coords[:,0])
        rr_arr = tuple(cur_coords[:,1])
        cc_arr = tuple(cur_coords[:,2])
        imgs_labeled_out[ii_arr, rr_arr, cc_arr] = void_id

        # Minimum feature size
        if num_void_voxels >= min_feat_size:

            # The optimal surface should contain the same volume as the
            # original volume of voxels. As of now, this is constraint
            # is not enforced. To achieve this, see the examples in
            # "../make_vtk_models/".
            verts, faces, normals, vals = vapi.convert_voxels_to_surface(cur_img_fill,
                iso_level=127, scale_spacing=voxel_width_um, g_sigdev=0.8)

            surf_void = vapi.make_vtk_surf_mesh(verts, faces, vals, 
                smth_iter=1)

            surf_void.translate(tuple(trans_vec_xyz), inplace=True)

            surf_area_void = surf_void.area  # (um^2) surface area
            surf_vol_void = surf_void.volume # (um^3) volume

            # Calculate sphericity. Ensure consistent units here!
            sphericity = 1.46459188756*np.cbrt((6.0*surf_vol_void)**2) / surf_area_void
            
            voxel_void.point_data['Sphericity'] = sphericity*np.ones(
                voxel_void.n_points, dtype=np.float32)

            # Calculate the bounding box of the sub-stack
            bbox_search_flag = 4
            bbox_obj, lwt_arr, b_cent = bbox_search(cur_img, bbox_search_flag)
            b_cent = b_cent + trans_vec # Correct for the translation
            #b_cent = b_cent*voxel_width_um # Convert to [um]

            # L, W, and T of the rotated bounding box
            b_l_len = voxel_width_um*lwt_arr[2,0] # [um]
            b_w_len = voxel_width_um*lwt_arr[1,0] # [um]
            b_t_len = voxel_width_um*lwt_arr[0,0] # [um]

            # The unit vectors for L, W, and T in (img, row, col) index coordinates
            b_l_vec = np.array([lwt_arr[2,1], lwt_arr[2,2], lwt_arr[2,3]])
            b_w_vec = np.array([lwt_arr[1,1], lwt_arr[1,2], lwt_arr[1,3]])
            b_t_vec = np.array([lwt_arr[0,1], lwt_arr[0,2], lwt_arr[0,3]])

            temp_void_props_list = [
                void_id, #1
                voxel_width_um, #2
                num_void_voxels, #3
                void_vol, #4
                eqv_diam, #5
                sphericity, #6
                pore_cent[0], #7
                pore_cent[1], #8
                pore_cent[2], #9
                void_vtk_cent[0], #10
                void_vtk_cent[1], #11
                void_vtk_cent[2], #12
                b_cent[0], #13
                b_cent[1], #14
                b_cent[2], #15
                b_l_len, #16
                b_w_len, #17
                b_t_len, #18
                b_l_vec[0], #19
                b_l_vec[1], #20 
                b_l_vec[2], #21
                b_w_vec[0], #22
                b_w_vec[1], #23
                b_w_vec[2], #24
                b_t_vec[0], #25
                b_t_vec[1], #26
                b_t_vec[2], #27
                ]

        else: 
            # Was not a large enough void to characterize in this case.
            # The -999 values will be replaced in NaN in the .csv file
            sphericity = -999
            voxel_void.point_data['Sphericity'] = np.zeros(
                voxel_void.n_points, dtype=np.float32)

            b_cent = np.array([-999, -999, -999])
            b_l_len = -999
            b_w_len = -999
            b_t_len = -999
            b_l_vec = np.array([-999, -999, -999])
            b_w_vec = np.array([-999, -999, -999])
            b_t_vec = np.array([-999, -999, -999])

            temp_void_props_list = [
                void_id, #1
                voxel_width_um, #2
                num_void_voxels, #3
                void_vol, #4
                eqv_diam, #5
                sphericity, #6
                pore_cent[0], #7
                pore_cent[1], #8
                pore_cent[2], #9
                void_vtk_cent[0], #10
                void_vtk_cent[1], #11
                void_vtk_cent[2], #12
                b_cent[0], #13
                b_cent[1], #14
                b_cent[2] , #15
                b_l_len, #16
                b_w_len, #17
                b_t_len, #18
                b_l_vec[0], #19
                b_l_vec[1], #20 
                b_l_vec[2], #21
                b_w_vec[0], #22
                b_w_vec[1], #23
                b_w_vec[2], #24
                b_t_vec[0], #25
                b_t_vec[1], #26
                b_t_vec[2], #27
                ]

        void_props.append(temp_void_props_list)
        vtk_list_pores.append(voxel_void)

        # Provide updates to the user
        if (void_index+1)%20 == 0:
            print(f"  Processed {void_index+1}/{n_feats} pores...")


    # -------- SAVE THE LABELED PORES --------

    # First, the image stack
    imex.save_image_seq(imgs_labeled_out, imgs_pores_dir_out_path, 
        imgs_name_out_substr)

    # Next, save the CSV file of void properties
    print(f"\nWriting pore properties to: {csv_pore_path_out}")
    save_2d_csv_file(void_props, void_props_fmt_str, csv_pore_path_out,
        void_props_header_str)


    # -------- CREATE THE VTK MODELS --------

    # Merge the VTK pore voxel-models into one
    if vtk_list_pores:
        print(f"\nMaking VTK model of labeled voids...")
        num_vtk = len(vtk_list_pores)
    
        if num_vtk > 1:
            merged_vtk = vtk_list_pores[0]
            for vtk_ii in np.arange(1, num_vtk):
                cur_vtk = vtk_list_pores[vtk_ii]
                merged_vtk.merge(cur_vtk, merge_points=False, inplace=True)
                
                if (vtk_ii+1)%20 == 0:
                    print(f"  Processed {vtk_ii+1}/{num_vtk} VTK files...")
        else:
            merged_vtk = vtk_list_pores[0]

        print(f"\nSaving labeled VTK pores to: {vtk_pore_path_out}")
        merged_vtk.save(vtk_pore_path_out)

    # Convert the voxel model of the solid to a surface mesh
    verts, faces, normals, vals = vapi.convert_voxels_to_surface(imgs_solid_bin,
        iso_level=127, scale_spacing=voxel_width_um, g_sigdev=0.8)

    surf_metal = vapi.make_vtk_surf_mesh(verts, faces, vals, 
        smth_iter=1)

    surf_metal.flip_y(point=[0.0, 0.0, 0.0], inplace=True)
    surf_metal.flip_normals()

    surf_metal.save(vtk_solid_path_out)

print(f"\nSCRIPT FINISHED SUCCESSFULLY!")





