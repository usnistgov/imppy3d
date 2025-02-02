# Import external dependencies
import numpy as np
from skimage import restoration as rest
from skimage import segmentation as seg
from skimage import filters as filt
from skimage import morphology as morph
import skimage.feature as sfeature
from skimage.util import img_as_bool, img_as_ubyte, img_as_float

# Import local packages
from . import ski_interactive_processing as ifun


def interact_driver_skeletonize(img_in, fltr_name_in="scikit"):
    """
    Binarizes an image by thinning connected pixels until they are just
    one pixel wide, also known as skeletonization. The implementation
    is done by SciKit-Image: skimage.morphology.skeletonize(). This is
    an interactive function that enables the user to change the
    parameters of the filter and see the results, thanks to
    the "widgets" available in Matplotlib.

    ---- INPUT ARGUMENTS ---- 
    [img_in]: Numpy array for a binary image. It is assumed that the
        image is already segmented and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

    fltr_name_in: A string that represents the name of the skeletonize
        algorithm to be applied. Currently, only one exists, named
        "scikit":

            "scikit": Uses functions from SciKit-Image to skeletonize
            the binary image.

    ---- RETURNED ----
    [img_out]: Returns the final image after closing the interactive
        session. img_out is in the same data type as img_in.

    ---- SIDE EFFECTS ----
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.

    """
    
    # ---- Local Copies ----
    fltr_name = fltr_name_in.lower()

    # List of strings of supported filter types
    fltr_list = ["scikit"]

    # Ensure that the filter name is recognized
    if fltr_name not in fltr_list:
        print(f"\n{fltr_name_in} is not a supported algorithm for skeletonizing an image."\
            f"Supported algorithms are: \n  {fltr_list}"\
            f"\nDefaulting to Scikit-Image's skeletonization.")
        fltr_name = "scikit"

    if fltr_name == "scikit":
        [img_fltr, fltr_params] = ifun.interact_skeletonize(img_in)
    
    # Using this function just to write to standard output
    apply_driver_skeletonize(img_in, fltr_params)

    return img_fltr


def apply_driver_skeletonize(img_in, fltr_params_in, quiet_in=False):
    """
    Binarizes an image by thinning connected pixels until they are just
    one pixel wide, also known as skeletonization. The implementation
    is done by SciKit-Image: skimage.morphology.skeletonize(). This is
    an non-interactive function related to the above function,
    interact_driver_skeletonize().

    ---- INPUT ARGUMENTS ---- 
    [img_in]: Numpy array for a binary image. It is assumed that the
        image is already segmented and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

    fltr_params_in: A list of parameters needed to perform the 
        skeletonize operation. The first parameter is a string, which
        determines what type of skeletonize algorithm to be applied, as
        well as the definitions of the remaining parameters. Currently,
        the only option is "scikit":

            ["scikit", apply_skel]
                apply_skel: A boolean which applies the skeletonize 
                    algorithm if True, and does not modify the image
                    if False.

    quiet_in: A boolean that determines if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing.

    ---- RETURNED ----
    [img_out]: Returns the resultant Numpy image after performing the
        image processing procedures. img_out is in the same data type as
        img_in.

    ---- SIDE EFFECTS ----
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. Strings are 
    printed to standard output.
    """

    # ---- Local Copies ----
    img = img_in.copy()
    fltr_params = fltr_params_in
    quiet = quiet_in

    fltr_name = (fltr_params[0]).lower()

    if fltr_name == "scikit":
        apply_skel = fltr_params[1]

        if apply_skel:
            img_bool = img_as_bool(img)
            img_temp = morph.skeletonize(img_bool)
            img_fltr = img_as_ubyte(img_temp)

        else:
            img_fltr = img

        if not quiet:
            print(f"\nSuccessfully skeletonized the image:\n"\
                f"    Applied skeletonize: {apply_skel}")

    return img_fltr


def interact_driver_del_features(img_in, fltr_name_in="scikit"):
    """
    Interactively delete features, and/or fill holes, of a specified
    size (i.e., area). This is accomplished using SkiKit-Image:
    skimage.morphology.remove_small_holes() and
    skimage.morphology.remove_small_objects(). This is an interactive
    function that enables the user to change the parameters of the
    filter and see the results, thanks to the "widgets" available in
    Matplotlib.

    ---- INPUT ARGUMENTS ---- 
    [img_in]: Numpy array for a binary image. It is assumed that the
        image is already segmented and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

    fltr_name_in: A string that represents the name of the deletion
        algorithm to be applied. Currently, only one exists, named
        "scikit": 

            "scikit": Uses functions from SciKit-Image to label and
            delete features of a specified size. Also uses functions
            from SciKit-Image to fill in holes of a specified size.
            1-connectivity is used in all cases.

    ---- RETURNED ----
    [img_out]: Returns the final image after closing the interactive
        session. img_out is in the same data type as img_in.

    ---- SIDE EFFECTS ----
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """

    # ---- Local Copies ----
    fltr_name = fltr_name_in.lower()

    # List of strings of supported filter types
    fltr_list = ["scikit"]

    # Ensure that the filter name is recognized
    if fltr_name not in fltr_list:
        print(f"\n{fltr_name_in} is not a supported algorithm for deleting features."\
            f"Supported algorithms are: \n  {fltr_list}"\
            f"\nDefaulting to Scikit-Image's feature deletion.")
        fltr_name = "scikit"

    if fltr_name == "scikit":
        [img_fltr, fltr_params] = ifun.interact_del_features(img_in)

    # Using this function just to write to standard output
    apply_driver_del_features(img_in, fltr_params)

    return img_fltr


def apply_driver_del_features(img_in, fltr_params_in, quiet_in=False):
    """
    Delete features and/or fill holes of a specified size (i.e., area).
    This is accomplished using the SkiKit-Image library. This is the
    non-interactive version of interact_driver_del_features() defined
    above.

    ---- INPUT ARGUMENTS ---- 
    [img_in]: Numpy array for a binary image. It is assumed that the
        image is already segmented and of type uint8. The array should
        thus be 2D, where each value represents the intensity for each
        corresponding pixel.   

    fltr_params_in: A list of parameters needed to perform the fill and
        deletion operations. The first parameter is a string, which
        determines what type of feature detection algorithms to use, as
        well as the definitions of the remaining parameters. Currently,
        only the "scikit" implementation is complete:

            ["scikit", max_hole_sz, min_feat_sz]
                max_hole_sz: The maximum area, in pixels, of a 
                    contiguous hole that will be filled. 1-connectivity
                    is assumed, and this should be an integer.
                min_feat_sz: The smallest allowable object size, in
                    pixels, assuming 1-connectivity. This should be an
                    integer.

    quiet_in: A boolean that determines if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing.

    ---- RETURNED ----
    [img_out]: Returns the resultant Numpy image after performing the
        image processing procedures. img_out is in the same data type as
        img_in.

    ---- SIDE EFFECTS ----
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. Strings are 
    printed to standard output.
    """

    # ---- Local Copies ----
    img = img_in.copy()
    fltr_params = fltr_params_in
    quiet = quiet_in

    fltr_name = (fltr_params[0]).lower()

    if fltr_name == "scikit":
        max_hole_sz = fltr_params[1]
        min_feat_sz = fltr_params[2]

        img_bool = img_as_bool(img)
        img_temp = morph.remove_small_holes(img_bool, max_hole_sz, connectivity=1)
        img_fltr = morph.remove_small_objects(img_temp, min_feat_sz, connectivity=1)
        img_fltr = img_as_ubyte(img_fltr)

        if not quiet:
            print(f"\nSuccessfully removed small holes and small objects:\n"\
                f"    Minimum Allowable Feature Size: {min_feat_sz}\n"\
                f"    Maximum Allowable Hole Size: {max_hole_sz}")

    return img_fltr


def interact_driver_edge_detect(img_in, fltr_name_in):
    """
    Enhances the edges of an image which are detected using the Canny
    edge filter, implemented by SciKit-Image. This is an interactive
    function that enables the user to change the parameters of the
    filter and see the results, thanks to the "widgets" available in
    Matplotlib.

    ---- INPUT ARGUMENTS ----
    [img_in]: Numpy array for a grayscale image. It is assumed that the
        image is already grayscale and of type uint8. The array should
        thus be 2D, where each value represents the intensity for each
        corresponding pixel.
    
    fltr_name_in: A string that represents the name of the thresholding
        filter to be applied. Can be either: 

            "canny": Applies the Canny edge detection algorithm.

    ---- RETURNED ----
    [img_fltr]: Returns the final image after closing the interactive
        session. img_fltr is in the same format as img_in.

    [img_mask]: A binary image where white pixels represent the edges
        found by the Canny algorithm.

    ---- SIDE EFFECTS ----
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """
    
    # ---- Local Copies ----
    fltr_name = fltr_name_in.lower()

    # List of strings of supported filter types
    fltr_list = ["canny"]

    # Ensure that the filter name is recognized
    if fltr_name not in fltr_list:
        print(f"\n{fltr_name_in} is not a supported edge detection. Supported "\
            f"edge detection algorithms are: \n  {fltr_list}"\
            "\nDefaulting to canny edge detection.")
        fltr_name = "canny"

    if fltr_name == "canny":
        [img_fltr, img_mask, fltr_params] = ifun.interact_canny_edge(img_in)

    # Using this function just to write to standard output
    apply_driver_edge_detect(img_in, fltr_params)

    return img_fltr, img_mask


def apply_driver_edge_detect(img_in, fltr_params_in, quiet_in=False):
    """
    Enhances the edges of an image which are detected using the Canny
    edge filter, implemented by SciKit-Image. This is the
    non-interactive version of interact_driver_edge_detect().

    ---- INPUT ARGUMENTS ----
    [img_in]: Numpy array for a grayscale image. It is assumed that the
        image is already grayscale and of type uint8. The array should
        thus be 2D, where each value represents the intensity for each
        corresponding pixel.

    fltr_params_in: A list of parameters needed to perform the edge
        detection. The first parameter is a string, which determines
        what type of edge detection filter to be applied, as well as the 
        definitions of the remaining parameters. Only the "canny" option
        is implemented at the moment:

            ["canny", sigma_out, low_thresh_out, high_thresh_out]
                sigma_out: Standard deviation of the Gaussian filter, 
                    which should be a float.
                low_thresh_out: Lower bound for hysteresis thresholding
                    (linking edges), which should be a float.
                high_thresh_out: Upper bound for hysteresis thresholding
                    (linking edges), which should be a float.

    quiet_in: A boolean that determines if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing.

    ---- RETURNED ----
    [img_fltr]: Returns the final image after closing the interactive
        session. img_fltr is in the same format as img_in.

    [img_mask]: A binary image where white pixels represent the edges
        found by the Canny algorithm.

    ---- SIDE EFFECTS ----
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. Strings are 
    printed to standard output.
    """

    # ---- Local Copies ----
    img = img_in.copy()
    fltr_params = fltr_params_in
    quiet = quiet_in

    fltr_name = (fltr_params[0]).lower()

    if fltr_name == "canny":
        sigma_out = fltr_params[1]
        low_thresh_out = fltr_params[2] 
        high_thresh_out = fltr_params[3]

        img_mask = sfeature.canny(img, sigma=sigma_out, 
            low_threshold=low_thresh_out, high_threshold=high_thresh_out)

        mask_out = img_as_ubyte(img_mask)
        img_fltr = img.copy()
        img_fltr[img_mask] = img[img_mask]/2

        if not quiet:
            print(f"\nSuccessfully applied the 'canny' edge detection:\n"\
                f"    Sigma: {sigma_out}\n"\
                f"    Low Threshold: {low_thresh_out}\n"\
                f"    High Threshold: {high_thresh_out}")

    return img_fltr, mask_out


def interact_driver_thresholding(img_in, fltr_name_in):
    """
    Interactively implements a thresholding operation in order to
    segment (or binarize) a UINT8 image into black (0) and white
    (255) pixels. The output parameters of this function are suitable
    for input for apply_driver_thresholding() below.

    ---- INPUT ARGUMENTS ----
    [img_in]: Numpy array for a grayscale image. It is assumed that the
        image is already grayscale and of type uint8. The array should
        thus be 2D, where each value represents the intensity for each
        corresponding pixel.
    
    fltr_name_in: A string that represents the name of the thresholding
        filter to be applied. Can be either: 
            
            "hysteresis_threshold_slider": Applies a threshold to an 
            image using hysteresis thresholding, which considers the
            connectivity of features. This interactive GUI utilizes
            slider widgets for input.
            
            "hysteresis_threshold_text": Applies a threshold to an 
            image using hysteresis thresholding, which considers the
            connectivity of features. This interactive GUI utilizes
            text box widgets for input.

            "adaptive_threshold": Applies adaptive thresholding, also
            known as local thresholding.

            "global_threshold": Uses a single value of grayscale 
            intensity as the thresholding criterion for the image.

    ---- RETURNED ----
    [img_out]: Returns the final image after closing the interactive
        session. img_out is in the same format as img_in.

    ---- SIDE EFFECTS ----
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """

    # ---- Local Copies ----
    fltr_name = fltr_name_in.lower()

    # List of strings of supported filter types
    fltr_list = ["hysteresis_threshold_slider", "hysteresis_threshold_text",
        "global_threshold"]

    # Ensure that the filter name is recognized
    if fltr_name not in fltr_list:
        print(f"\n{fltr_name_in} is not a supported threshold type. Supported "\
            f"threshold types are: \n  {fltr_list}"\
            "\nDefaulting to hysteresis thresholding.")
        fltr_name = "hysteresis_threshold_slider"

    # Filter Type: "hysteresis_threshold" for slider inputs
    if fltr_name == "hysteresis_threshold_slider":
        [img_fltr, fltr_params] = ifun.interact_hysteresis_threshold(img_in)

    # Filter Type: "hysteresis_threshold2" for text inputs
    elif fltr_name == "hysteresis_threshold_text":
        [img_fltr, fltr_params] = ifun.interact_hysteresis_threshold2(img_in)

    elif fltr_name == "global_threshold":
        [img_fltr, fltr_params] = ifun.interact_global_thresholding(img_in)

    elif fltr_name == "adaptive_threshold":
        [img_fltr, fltr_params] = ifun.interact_adaptive_thresholding(img_in)

    # Using this function just to write to standard output
    apply_driver_thresholding(img_in, fltr_params)

    return img_fltr


def apply_driver_thresholding(img_in, fltr_params_in, quiet_in=False):
    """
    Applies a thresholding operation in order to segment (or binarize) a
    UINT8 image into black (0) and white(255) pixels. This is the
    non-interactive version of interact_driver_thresholding().

    ---- INPUT ARGUMENTS ----
    [img_in]: Numpy array for a grayscale image. It is assumed that the
        image is already grayscale and of type uint8. The array should
        thus be 2D, where each value represents the intensity for each
        corresponding pixel.
    
    fltr_params_in: A list of parameters needed to perform the threshold
        operation. The first parameter is a string, which determines
        what type of threshold filter to be applied, as well as the 
        definitions of the remaining parameters. Available options are: 
            
            ["hysteresis_threshold", low_val_out, high_val_out] 
                low_val_out: Lower threshold intensity as an integer
                high_val_out: Upper threshold intensity as an integer

            ["adaptive_threshold", block_sz, thresh_offset]
                block_sz: Odd size of pixel neighborhood which is used
                    to calculate the threshold value Should be an
                    integer.
                thresh_offset: Constant subtracted from weighted mean of
                    neighborhood to calculate the local threshold value.
                    Default offset is 0.0, and it is treated as a float.

            ["global_threshold", global_thresh]
                global_thresh: A grayscale intensity to use as a 
                    criterion for thresholding. Values greater than 
                    this threshold will become white. Values less than
                    this threshold will become black.
            
    quiet_in: A boolean that determines if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing.

    ---- RETURNED ----
    [img_out]: Returns the resultant Numpy image after performing the
        image processing procedures. img_out is in the same format as
        img_in.

    ---- SIDE EFFECTS ----
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. Strings are 
    printed to standard output.
    """

    # ---- Local Copies ----
    img = img_in.copy()
    fltr_params = fltr_params_in
    quiet = quiet_in

    fltr_name = (fltr_params[0]).lower()

    if fltr_name == "hysteresis_threshold":
        low_val_in = fltr_params[1]
        high_val_in = fltr_params[2]

        img_temp = filt.apply_hysteresis_threshold(img, low_val_in, 
            high_val_in)
        img_fltr = img_as_ubyte(img_temp)
        
        if not quiet:
            print(f"\nSuccessfully applied the 'hysteresis_threshold':\n"\
                f"    Lower grayscale intensity limit: {low_val_in}\n"\
                f"    Upper grayscale intensity limit: {high_val_in}")

    elif fltr_name == "global_threshold":
        global_thresh = fltr_params[1]
        img_fltr = img_as_ubyte(img > global_thresh)

        if not quiet:
            print(f"\nSuccessfully applied the 'global_threshold':\n"\
                f"    Global threshold value: {global_thresh}")

    elif fltr_name == "adaptive_threshold":
        block_sz = fltr_params[1]
        thresh_offset = fltr_params[2]

        if block_sz % 2 == 0:
            block_sz = block_sz + 1

        img_thresh = filt.threshold_local(img, block_size=block_sz, 
            method='gaussian', offset=thresh_offset)

        img_fltr = img_as_ubyte(img > img_thresh)

        if not quiet:
            print(f"\nSuccessfully applied the 'adaptive_threshold':\n"\
                f"    Block Size: {block_sz}\n"\
                f"    Offset (+/-): {thresh_offset}")

    else:
        print(f"\nERROR: {fltr_name} is not a supported sharpening filter.")
        img_fltr = img

    return img_fltr


def interact_driver_sharpen(img_in, fltr_name_in):
    """
    Interactively implements a sharpening filter. The output parameters
    of this function are suitable for input for apply_driver_sharpen()
    below.

    ---- INPUT ARGUMENTS ----
    [img_in]: Numpy array for a grayscale image. It is assumed that the
        image is already grayscale and of type uint8. The array should
        thus be 2D, where each value represents the intensity for each
        corresponding pixel.
    
    fltr_name_in: A string that represents the name of the sharpening
        filter to be applied. Currently only "unsharp_mask" is 
        supported.  
            
            "unsharp_mask":  Sharpens an image based on an unsharp
                mask using a Ski-Image function. An unsharp mask is
                based on a weighted addition between  the original
                image and Gaussian blurred version. 

    ---- RETURNED ----
    [img_out]: Returns the final image after closing the interactive
        session. img_out is in the same format as img_in.

    ---- SIDE EFFECTS ----
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """

    # ---- Local Copies ----
    fltr_name = fltr_name_in.lower()

    # List of strings of supported filter types
    fltr_list = ["unsharp_mask"]

    # Ensure that the filter name is recognized
    if fltr_name not in fltr_list:
        print(f"\n{fltr_name_in} is not a supported filter type. Supported "\
            f"filter types are: \n  {fltr_list}"\
            "\nDefaulting to unsharp mask.")
        fltr_name = "unsharp_mask"

    # -------- Filter Type: "unsharp_mask" --------
    if fltr_name == "unsharp_mask":
        [img_fltr, fltr_params] = ifun.interact_unsharp_mask(img_in)

    # Using this function just to write to standard output
    apply_driver_sharpen(img_in, fltr_params)

    return img_fltr


def apply_driver_sharpen(img_in, fltr_params_in, quiet_in=False):
    """
    Applies a sharpening filter. This is the non-interactive version of
    interact_driver_sharpen().

    ---- INPUT ARGUMENTS ----
    [img_in]: Numpy array for a grayscale image. It is assumed that the
        image is already grayscale and of type uint8. The array should
        thus be 2D, where each value represents the intensity for each
        corresponding pixel.
    
    fltr_params_in: A list of parameters needed to perform the 
        sharpening operation. The first parameter is a string, which
        determines what type of sharpen filter to be applied, as well
        as the definitions of the remaining parameters. As of now,
        fltr_params_in[0] must be "unsharp_mask". Example parameter
        lists are given below for each type, 
            
            ["unsharp_mask", radius_out, amount_out] 

                radius_out: Radius of the kernel for the unsharp filter. 
                    If zero, then no filter was applied. Should be an
                    integer.

                amount_out: The sharpening details will be amplified 
                    with this factor, which can be a negative or 
                    positive float.
            
    quiet_in: A boolean that determines if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing.
            
    NOTE: Currently, only sharpening via an unsharp mask has been
    implemented for thresholding from the Skimage library. For
    additional methods, see the OpenCV interactive scripts in
    cv_driver_functions.py.

    ---- RETURNED ----
    [img_out]: Returns the resultant Numpy image after performing the
        image processing procedures. img_out is in the same format as
        img_in.

    ---- SIDE EFFECTS ----
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. Strings are 
    printed to standard output.
    """

    # ---- Local Copies ----
    img = img_in.copy()
    fltr_params = fltr_params_in
    quiet = quiet_in

    fltr_name = (fltr_params[0]).lower()

    if fltr_name == "unsharp_mask":
        radius_in = fltr_params[1]
        amount_in = fltr_params[2]

        img_temp = filt.unsharp_mask(img, radius=radius_in, 
            amount=amount_in, channel_axis=None)
        img_fltr = img_as_ubyte(img_temp)
        
        if not quiet:
            print(f"\nSuccessfully applied the 'unsharp_mask' sharpening filter:\n"\
                f"    Radius of sharpening kernel: {radius_in}\n"\
                f"    Amount of sharpening: {amount_in}")

    else:
        print(f"\nERROR: {fltr_name} is not a supported sharpening filter.")
        img_fltr = img

    return img_fltr


def interact_driver_denoise(img_in, fltr_name_in):
    """
    Interactively implements a denoising filter. The output parameters
    of this function are suitable for input for apply_driver_denoise()
    below.

    ---- INPUT ARGUMENTS ----
    [img_in]: Numpy array for a grayscale image. It is assumed that the
        image is already grayscale and of type uint8. The array should
        thus be 2D, where each value represents the intensity for each
        corresponding pixel.
    
    fltr_name_in: A string that represents the name of the denoising
        filter to be applied. Currently, "tv_chambolle" or "nl_means" 
        are supported.  
            
            "tv_chambolle": Perform total-variation denoising on an 
                image. The principle of total variation denoising is to
                minimize the total variation of the image, which can be
                roughly described as the integral of the norm of the
                image gradient. Total variation denoising tends to
                produce “cartoon-like” images, that is, 
                piecewise-constant images.

            "nl_means": Perform non-local means denoising on an image. 
                The non-local means algorithm is well suited for
                denoising images with specific textures. The principle
                of the algorithm is to average the value of a given
                pixel with values of other pixels in a limited
                neighbourhood, provided that the patches centered on
                the other pixels are similar enough to the patch
                centered on the pixel of interest.

    ---- RETURNED ----
    [img_out]: Returns the final image after closing the interactive
        session. img_out is in the same format as img_in.

    ---- SIDE EFFECTS ----
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """

    # ---- Local Copies ----
    fltr_name = fltr_name_in.lower()

    # List of strings of supported filter types
    fltr_list = ["tv_chambolle", "nl_means"]

    # Ensure that the filter name is recognized
    if fltr_name not in fltr_list:
        print(f"\n{fltr_name_in} is not a supported filter type. Supported "\
            f"filter types are: \n  {fltr_list}"\
            "\nDefaulting to non-local means denoising.")
        fltr_name = "nl_means"

    # -------- Filter Type: "nl_means" --------
    if fltr_name == "nl_means":
        [img_fltr, fltr_params] = ifun.interact_nl_means_denoise(img_in)

    # -------- Filter Type: "tv_chambolle" --------
    elif fltr_name == "tv_chambolle":
        [img_fltr, fltr_params] = ifun.interact_tv_denoise(img_in)

    # Using this function just to write to standard output
    apply_driver_denoise(img_in, fltr_params)

    return img_fltr


def apply_driver_denoise(img_in, fltr_params_in, quiet_in=False):
    """
    Applies a denoising filter. This is the non-interactive version of
    interact_driver_denoise().

    ---- INPUT ARGUMENTS ----
    [img_in]: Numpy array for a grayscale image. It is assumed that the
        image is already grayscale and of type uint8. The array should
        thus be 2D, where each value represents the intensity for each
        corresponding pixel.
    
    fltr_params_in: A list of parameters needed to perform the 
        denoising operation. The first parameter is a string, which
        determines what type of denoise filter to be applied, as well
        as the definitions of the remaining parameters. As of now,
        fltr_params_in[0] must be either "nl_means" or "tv_chambolle". 
        Example parameter lists are given below for each type, 
            
            ["tv_chambolle", weight_out, eps_out, n_iter_max_out]

                weight_out: Denoising weight. The greater weight, the 
                    more denoising (at the expense of fidelity)

                eps_out: Relative difference of the value of the cost
                    function that determines the stop criterion. See the
                    Skimage documentation for additional details.

                n_iter_max_out: Maximal number of iterations used for 
                    the optimization.

            ["nl_means", h_out, patch_size_out, patch_dist_out]

                h_out: Cut-off distance (in gray levels). The higher
                    h_out, the more permissive one is in accepting
                    patches. A higher h_out results in a smoother
                    image, at the expense of blurring features. For a
                    Gaussian noise of standard deviation sigma, a rule
                    of thumb is to choose the value of h_out to be
                    on the same order of magnitude as the standard 
                    deviation of the Gaussian noise. This is a float.

                patch_size_out: Size, in pixels, of the patches used for
                    denoising. Should be an integer.

                patch_dist_out: Maximal distance, in pixels, where to 
                    search for patches used for denoising.
            
    quiet_in: A boolean that determines if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing.

    ---- RETURNED ----
    [img_out]: Returns the resultant Numpy image after performing the
        image processing procedures. img_out is in the same format as
        img_in.

    ---- SIDE EFFECTS ----
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. Strings are 
    printed to standard output.
    """

    # ---- Local Copies ----
    img = img_in.copy()
    fltr_params = fltr_params_in
    quiet = quiet_in

    fltr_name = (fltr_params[0]).lower()

    if fltr_name == "nl_means":
        h_in = fltr_params[1]
        patch_size_in = fltr_params[2]
        patch_dist_in = fltr_params[3]

        sig_noise = rest.estimate_sigma(img_as_float(img), 
            average_sigmas=True, channel_axis=None)

        img_fltr = rest.denoise_nl_means(img, h=h_in, 
            sigma=sig_noise, fast_mode=True, patch_size=patch_size_in, 
            patch_distance=patch_dist_in, channel_axis=None)

        img_fltr = img_as_ubyte(img_fltr)
        
        if not quiet:
            print(f"\nSuccessfully applied the 'nl_means' denoising filter:\n"\
                f"    'h' (cut-off range in gray levels): {h_in}\n"\
                f"    Patch size: {patch_size_in}\n"\
                f"    Maximum search distance: {patch_dist_in}" )

    elif fltr_name == "tv_chambolle":
        weight_in = fltr_params[1]
        eps_in = fltr_params[2]
        n_iter_max_in = fltr_params[3]

        img_fltr = rest.denoise_tv_chambolle(img, weight=weight_in,
            eps=eps_in, n_iter_max=n_iter_max_in, multichannel=False, 
            channel_axis=None)

        img_fltr = img_as_ubyte(img_fltr)

        if not quiet:
            print(f"\nSuccessfully applied the 'tv_chambolle' denoising filter:\n"\
                f"    Denoising weight: {weight_in}\n"\
                f"    EPS stop criterion: {eps_in}\n"\
                f"    Maximal number of iterations: {n_iter_max_in}" )

    else:
        print(f"\nERROR: {fltr_name} is not a supported denoise filter.")
        img_fltr = img

    return img_fltr


def interact_driver_ridge_filter(img_in, fltr_name_in, flood_ext_in=True):
    """
    Interactively implements a "ridge" filter. This filter can be used
    to detect continuous ridges, e.g. tubes, wrinkles, rivers. This
    type of filter is also referred to as a ridge operator. The output
    parameters of this function are suitable for input for
    apply_driver_ridge_filter() below.

    ---- INPUT ARGUMENTS ----
    [img_in]: Numpy array for a grayscale image. It is assumed that the
        image is already grayscale and of type uint8. The array should
        thus be 2D, where each value represents the intensity for each
        corresponding pixel.
    
    fltr_name_in: A string that represents the name of the ridge
        filter to be applied. Currently, only "sato_tubeness" is
        supported.  
            
            "sato_tubeness": Filter an image with the Sato tubeness 
                filter. Calculates the eigenvectors of the Hessian to
                compute the similarity of an image region to tubes,
                according to the method described in Sato et al. (1998)
                "Three-dimensional multi-scale line filter for 
                segmentation and visualization of curvilinear structures
                in medical images" DOI:10.1016/S1361-8415(98)80009-1

    ---- RETURNED ----
    [img_out]: Returns the final image after closing the interactive
        session. img_out is in the same format as img_in.

    ---- SIDE EFFECTS ----
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """

    # ---- Local Copies ----
    fltr_name = fltr_name_in.lower()
    img_0 = img_in.copy()

    # List of strings of supported filter types
    fltr_list = ["sato_tubeness"]

    # Ensure that the filter name is recognized
    if fltr_name not in fltr_list:
        print(f"\n{fltr_name_in} is not a supported filter type. Supported "\
            f"filter types are: \n  {fltr_list}"\
            "\nDefaulting to Sato tubeness.")
        fltr_name = "sato_tubeness"

    # Apply a flood-fill operation to the exterior region
    if flood_ext_in:
        flood_val = np.mean(img_0[img_0>0])
        flood_val = (np.round(flood_val)).astype(np.uint16)

        img_temp = np.pad(img_0, 1, mode='constant', 
            constant_values=img_0[0,0])

        img_temp = img_as_ubyte(img_temp)

        img_temp = seg.flood_fill(img_temp, (0,0), flood_val, 
            connectivity=1)

        img_0 = img_temp[1:-1, 1:-1]

    # -------- Filter Type: "sato_tubeness" --------
    if fltr_name == "sato_tubeness":
        [img_fltr, fltr_params] = ifun.interact_sato_tubeness(img_0)

    # Using this function just to write to standard output
    apply_driver_ridge_filter(img_0, fltr_params)

    return img_fltr


def apply_driver_ridge_filter(img_in, fltr_params_in, flood_ext_in=True,
    quiet_in=False):
    """
    Applies a "ridge" filter. This filter can be used to detect
    continuous ridges, e.g. tubes, wrinkles, rivers. This type of
    filter is also referred to as a ridge operator. This is the
    non-interactive version of interact_driver_ridge_filter().

    ---- INPUT ARGUMENTS ----
    [img_in]: Numpy array for a grayscale image. It is assumed that the
        image is already grayscale and of type uint8. The array should
        thus be 2D, where each value represents the intensity for each
        corresponding pixel.
    
    fltr_params_in: A list of parameters needed to perform the 
        ridge filter. The first parameter is a string, which
        determines what type of ridge filter to be applied, as well
        as the definitions of the remaining parameters. As of now,
        fltr_params_in[0] must be "sato_tubeness". 
        Example parameter lists are given below for each type, 
            
            ["sato_tubeness", mask_val_out, sig_max_out, blk_ridges_out]

                mask_val_out: Only values greater than mask_val_out will
                    be altered by this filter. Hence, this integer acts
                    as a basic mask.

                sig_max_out: The maximum sigma used to scale  the 
                    filter. A range of sigma values will automatically
                    creates from one to sig_max_out in steps of one,
                    which is then utilized by the Sato filter. See the
                    Skimage documentation for more details. Typically
                    a value of ten is appropriate.

                blk_ridges_out: A boolean that affects whether the 
                    filter should detect black ridges or white ridges.
                    When True, the filter detects black ridges.
            
    quiet_in: A boolean that determines if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing.

    ---- RETURNED ----
    [img_out]: Returns the resultant Numpy image after performing the
        image processing procedures. img_out is in the same format as
        img_in.

    ---- SIDE EFFECTS ----
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. Strings are 
    printed to standard output.
    """

    # ---- Local Copies ----
    img = img_in.copy()
    fltr_params = fltr_params_in
    quiet = quiet_in

    fltr_name = (fltr_params[0]).lower()

    # Apply a flood-fill operation to the exterior region
    if flood_ext_in:
        flood_val = np.mean(img[img>0])
        flood_val = (np.round(flood_val)).astype(np.uint16)

        img_temp = np.pad(img, 1, mode='constant', 
            constant_values=img[0,0])

        img_temp = img_as_ubyte(img_temp)

        img_temp = seg.flood_fill(img_temp, (0,0), flood_val, 
            connectivity=1)

        img = img_temp[1:-1, 1:-1]

    if fltr_name == "sato_tubeness":
        mask_val_in = fltr_params[1]
        sig_max_in = fltr_params[2]
        blk_ridges_in = fltr_params[3]

        mask = (img > mask_val_in)

        img_temp = filt.sato(img, sigmas=range(1, sig_max_in), 
            black_ridges=blk_ridges_in)*mask

        img_temp = img_temp/np.amax(img_temp)
        img_fltr = img_as_ubyte(img_temp)

        if not quiet:
            print(f"\nSuccessfully applied the 'sato_tubeness' denoising filter:\n"\
                f"    Lower masking limit of grayscale values: {mask_val_in}\n"\
                f"    Max sigma used as a scale: {sig_max_in}\n"\
                f"    Detect black ridges (boolean): {blk_ridges_in}" )

    else:
        print(f"\nERROR: {fltr_name} is not a supported ridge filter.")
        img_fltr = img

    return img_fltr


def interact_driver_morph(img_in):
    """
    Interactively applies a morphological operation on a binary
    (i.e., segmented) image. More specifically, apply either an
    erosion, dilation, "opening", or "closing" with the option of
    choosing different kernel shapes and sizes. The implementations of
    these operations are based on the Skimage library. The output
    parameters of this function are suitable for input for
    apply_driver_morph_3d() below.

    ---- INPUT ARGUMENTS ----
    [img_in]: Numpy array for a grayscale image. It is assumed that the
        image is already grayscale and of type uint8. The array should
        thus be 2D, where each value represents the intensity for each
        corresponding pixel.

    ---- RETURNED ----
    [img_out]: Returns the final image after closing the interactive
        session. img_out is in the same format as img_in.

    ---- SIDE EFFECTS ----
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """

    # ---- Local Copies ----
    img_0 = img_in.copy()

    # Run the interactive morphological operations GUI
    [img_fltr, fltr_params] = ifun.interact_binary_morph(img_0)

    # Run the driver application simply for the text output
    apply_driver_morph(img_0, fltr_params)

    return img_fltr


def apply_driver_morph(img_in, fltr_params_in, quiet_in=False):
    """
    Applies a morphological operation on a binary (i.e., segmented)
    image. More specifically, apply either an erosion,
    dilation, "opening", or "closing" with the option of choosing
    different kernel shapes and sizes. The implementations of these
    operations are based on the Skimage library. This is the 2D,
    non-interactive version of interact_driver_morph(). See 
    apply_driver_morph_3d() for a 3D version of this function.

    ---- INPUT ARGUMENTS ----
    [img_in]: Numpy array for a 2D grayscale image. It is 
        assumed that the image is already grayscale and of type uint8.
        Each value in the 2D array represents the intensity for each
        corresponding pixel.
    
    fltr_params_in: A list of parameters needed to perform the 
        morphological operation. The three input parameters for this
        list are described below, 
            
            [operation_type_out, footprint_type_out, n_radius_out]

                operation_type_out: An integer flag that determines what 
                    type of morphological operation to perform:

                        0: binary_closing
                        1: binary_opening
                        2: binary_dilation
                        3: binary_erosion

                footprint_type_out: An integer flag that determines what  
                    type of 2D neighborhood to use:

                        0: square 
                        1: disk 
                        2: diamond 

                n_radius_out: Radius of the footprint neighborhood in 
                    pixels, as an integer.
            
    quiet_in: A boolean that determines if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing.

    ---- RETURNED ----
    [img_out]: Returns the resultant Numpy image array after performing
        the image processing procedures. img_out is in the same format 
        as img_in.

    ---- SIDE EFFECTS ----
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. Strings are 
    printed to standard output.
    """

    # ---- Local Copies ----
    img_0 = img_in.copy()
    fltr_params = fltr_params_in
    quiet = quiet_in

    # Determine what type of morphological operation to perform
    # 0: binary_closing
    # 1: binary_opening
    # 2: binary_dilation
    # 3: binary_erosion
    operation_type = fltr_params[0]

    # Determine what type of 2D neighborhood to use.
    # 0: square (which corresponds to a cube in 3D)
    # 1: disk (which corresponds to a ball in 3D)
    # 2: diamond (which corresponds to a octahedron in 3D)
    footprint_type = fltr_params[1]

    # Radius of the footprint neighborhood (in pixels)
    n_radius = fltr_params[2]

    if not quiet:
        print(f"\nApplying morphological operation...")

    if footprint_type == 0:
        temp_width = 2*n_radius + 1
        temp_footprint = morph.square(temp_width)
        
        if not quiet:
            print(f"    Footprint type: 'square'")
            print(f"    Footprint radius (pixels): {n_radius}")

    elif footprint_type == 1:
        temp_footprint = morph.disk(n_radius)

        if not quiet:
            print(f"    Footprint type: 'disk'")
            print(f"    Footprint radius (pixels): {n_radius}")

    else: # footprint_type == 2
        temp_footprint = morph.diamond(n_radius)

        if not quiet:
            print(f"    Footprint type: 'diamond'")
            print(f"    Footprint radius (pixels): {n_radius}")

    if operation_type == 0:
        img_temp = morph.binary_closing(img_0, footprint=temp_footprint)

        if not quiet:
            print(f"    Operation type: 'binary_closing'")

    elif operation_type == 1:
        img_temp = morph.binary_opening(img_0, footprint=temp_footprint)

        if not quiet:
            print(f"    Operation type: 'binary_opening'")

    elif operation_type == 2:
        img_temp = morph.binary_dilation(img_0, footprint=temp_footprint)

        if not quiet:
            print(f"    Operation type: 'binary_dilation'")

    else: # operation_type == 3
        img_temp = morph.binary_erosion(img_0, footprint=temp_footprint)

        if not quiet:
            print(f"    Operation type: 'binary_erosion'")

    img_fltr = img_as_ubyte(img_temp)

    return img_fltr


def apply_driver_morph_3d(img_in, fltr_params_in, quiet_in=False):
    """
    Applies a morphological operation on a binary (i.e., segmented)
    image. More specifically, apply either an erosion,
    dilation, "opening", or "closing" with the option of choosing
    different kernel shapes and sizes. The implementations of these
    operations are based on the Skimage library. This is the 3D,
    non-interactive version of interact_driver_morph().

    ---- INPUT ARGUMENTS ----
    [img_in]: Numpy array for a 3D set of grayscale images. It is 
        assumed that the images are already grayscale and of type 
        uint8. Each value in the 3D array represents the intensity 
        for each corresponding pixel.
    
    fltr_params_in: A list of parameters needed to perform the 
        morphological operation. The three input parameters for this
        list are described below, 
            
            [operation_type_out, footprint_type_out, n_radius_out]

                operation_type_out: An integer flag that determines what 
                    type of morphological operation to perform:

                        0: binary_closing
                        1: binary_opening
                        2: binary_dilation
                        3: binary_erosion

                footprint_type_out: An integer flag that determines what  
                    type of 3D neighborhood to use:

                        0: cube 
                        1: ball 
                        2: octahedron 

                n_radius_out: Radius of the footprint neighborhood in 
                    pixels, as an integer.
            
    quiet_in: A boolean that determines if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing.

    ---- RETURNED ----
    [img_out]: Returns the resultant Numpy image array after performing
        the image processing procedures. img_out is in the same format 
        as img_in.

    ---- SIDE EFFECTS ----
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. Strings are 
    printed to standard output.
    """

    # ---- Local Copies ----
    img_0 = img_in.copy()
    fltr_params = fltr_params_in
    quiet = quiet_in

    # Determine what type of morphological operation to perform
    # 0: binary_closing
    # 1: binary_opening
    # 2: binary_dilation
    # 3: binary_erosion
    operation_type = fltr_params[0]

    # Determine what type of 3D neighborhood to use.
    # 0: cube
    # 1: ball
    # 2: octahedron
    footprint_type = fltr_params[1]

    # Radius of the footprint neighborhood (in pixels)
    n_radius = fltr_params[2]

    if not quiet:
        print(f"\nApplying morphological operation...")

    if footprint_type == 0:
        temp_width = 2*n_radius + 1
        temp_footprint = morph.cube(temp_width)
        
        if not quiet:
            print(f"    Footprint type: 'cube'")
            print(f"    Footprint radius (pixels): {n_radius}")

    elif footprint_type == 1:
        temp_footprint = morph.ball(n_radius)

        if not quiet:
            print(f"    Footprint type: 'ball'")
            print(f"    Footprint radius (pixels): {n_radius}")

    else: # footprint_type == 2
        temp_footprint = morph.octahedron(n_radius)

        if not quiet:
            print(f"    Footprint type: 'octahedron'")
            print(f"    Footprint radius (pixels): {n_radius}")

    if operation_type == 0:
        img_temp = morph.binary_closing(img_0, footprint=temp_footprint)

        if not quiet:
            print(f"    Operation type: 'binary_closing'")

    elif operation_type == 1:
        img_temp = morph.binary_opening(img_0, footprint=temp_footprint)

        if not quiet:
            print(f"    Operation type: 'binary_opening'")

    elif operation_type == 2:
        img_temp = morph.binary_dilation(img_0, footprint=temp_footprint)

        if not quiet:
            print(f"    Operation type: 'binary_dilation'")

    else: # operation_type == 3
        img_temp = morph.binary_erosion(img_0, footprint=temp_footprint)

        if not quiet:
            print(f"    Operation type: 'binary_erosion'")

    img_fltr = img_as_ubyte(img_temp)

    return img_fltr
