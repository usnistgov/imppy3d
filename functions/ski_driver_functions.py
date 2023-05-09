# Import external dependencies
import numpy as np
from skimage import restoration as rest
from skimage import segmentation as seg
from skimage import filters as filt
from skimage import morphology as morph
from skimage.util import img_as_ubyte, img_as_float

# Import local packages
import ski_processing_wrappers as wrap
import ski_interactive_processing as ifun

import plt_wrappers as pwrap
import matplotlib.pyplot as plt


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
        filter to be applied. Can be either 
        "hysteresis_threshold_slider" or "hysteresis_threshold_text". 
            
            "hysteresis_threshold_slider":  Applies a threshold to an 
            image using hysteresis thresholding, which considers the
            connectivity of features. This interactive GUI utilizes
            slider widgets for input.
            
            "hysteresis_threshold_text":  Applies a threshold to an 
            image using hysteresis thresholding, which considers the
            connectivity of features. This interactive GUI utilizes
            text box widgets for input.
            
    NOTE: Currently, only hysteresis thresholding has been implemented
    for thresholding from the Skimage library. For adaptive
    thresholding or conventional global thresholding, see the OpenCV
    interactive scripts in cv_driver_functions.py.

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
    fltr_list = ["hysteresis_threshold_slider", "hysteresis_threshold_text"]

    # Ensure that the filter name is recognized
    if fltr_name not in fltr_list:
        print(f"\n{fltr_name_in} is not a supported threshold type. Supported "\
            f"threshold types are: \n  {fltr_list}"\
            "\nDefaulting to hysteresis thresholding.")
        fltr_name = "hysteresis_threshold_slider"

    # -------- Filter Type: "hysteresis_threshold" for slider inputs --------
    if fltr_name == "hysteresis_threshold_slider":
        [img_fltr, fltr_params] = ifun.interact_hysteresis_threshold(img_in)

    # -------- Filter Type: "hysteresis_threshold2" for text inputs --------
    if fltr_name == "hysteresis_threshold_text":
        [img_fltr, fltr_params] = ifun.interact_hysteresis_threshold2(img_in)

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
        definitions of the remaining parameters. As of now, 
        fltr_params_in[0] must be "hysteresis_threshold". 
        Example parameter lists are given below for each type, 
            
            ["hysteresis_threshold", low_val_out, high_val_out] 

                low_val_out: Lower threshold intensity as an integer

                high_val_out: Upper threshold intensity as an integer
            
    quiet_in: A boolean that determines if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing.
            
    NOTE: Currently, only hysteresis thresholding has been implemented
    for thresholding from the Skimage library. For adaptive
    thresholding or conventional global thresholding, see the OpenCV
    interactive scripts in cv_driver_functions.py.

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
