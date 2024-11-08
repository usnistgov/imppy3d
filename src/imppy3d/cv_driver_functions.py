# Import external dependencies
import numpy as np
import cv2 as cv

# Import local packages
from . import cv_processing_wrappers as wrap
from . import cv_interactive_processing as ifun


def interact_driver_blur(img_in, fltr_name_in):
    """
    Interactively implements a blur filter on the provided image. The
    parameters for the image processing are controlled by trackbars in
    the interactive window, and the resultant effects are updated in
    real-time. Upon hitting the Enter or Esc keys, the interactive 
    window closes, and then this function returns the final image. Basic
    details of the supported blur filters are given below, but more
    details are available at "https://docs.opencv.org/4.2.0/d4/d13/
    tutorial_py_filtering.html". This function is related to 
    apply_driver_blur(...).

    ---- INPUT ARGUMENTS ----
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.
    
    fltr_name_in: A string that represents the name of the blur filter
        to be applied. Can be either "average", "gaussian", "median", or
        "bilateral".  
            
            "average": Equal-weighted averaging kernel
            
            "gaussian": Gaussian-weighted kernel
            
            "median": The center weight of kernal is the median value
            
            "bilateral": Edge-preserving Gaussian kernel

    ---- RETURNED ----
    [img_fltr]: Returns the final image after closing the interactive
        session. img_fltr is in the same format as img_in.

    ---- SIDE EFFECTS ----
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """

    # ---- Start Local Copies ----
    fltr_name = fltr_name_in.lower() # Ensure string is all lowercase
    # ---- End Local Copies ----

    # List of strings of the supported filter types
    fltr_list = ["average", "gaussian", "median", "bilateral"]

    # Ensure that the filter name is recognized
    if fltr_name not in fltr_list:
        print(f"\n{fltr_name_in} is not a supported filter type. Supported "\
            f"filter types are: {fltr_list}\nDefaulting to a Gaussian.")
        fltr_name = "gaussian"

    # -------- Filter Type: "average" --------
    if fltr_name == "average":
        [img_fltr, fltr_params] = ifun.interact_average_filter(img_in)

    # -------- Filter Type: "gaussian" --------
    elif fltr_name == "gaussian":
        [img_fltr, fltr_params] = ifun.interact_gaussian_filter(img_in)

    # -------- Filter Type: "median" --------
    elif fltr_name == "median":
        [img_fltr, fltr_params] = ifun.interact_median_filter(img_in)

    # -------- Filter Type: "bilateral" --------
    elif fltr_name == "bilateral":
        [img_fltr, fltr_params] = ifun.interact_bilateral_filter(img_in)
    
    # Using this function just to write to standard output
    apply_driver_blur(img_in, fltr_params)

    return img_fltr


def apply_driver_blur(img_in, fltr_params_in, quiet_in=False):
    """
    Applies a blur filter to the provided image based on the input
    parameters. This is the non-interactive version of the related
    function, interact_driver_blur(...). For more details about the
    supported blur filters, go to "https://docs.opencv.org/4.2.0/d4/
    d13/tutorial_py_filtering.html".

    ---- INPUT ARGUMENTS ----
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.
    
    [fltr_params_in]: A list of parameters needed to perform the blur
        operation. The first parameter is a string, which determines
        what type of blur filter to be applied, as well as the 
        definitions of the remaining parameters. fltr_params_in[0] can
        either be "average", "gaussian", "median", or "bilateral". 
        Example parameter lists are given below for each type,
            
            ["average", (k_size, k_size)]
                k_size: Kernel size for the average filter. If zero, 
                    then no filter was applied.
            
            ["gaussian", (k_size,k_size), std_dev]
                k_size: Kernel size for the Gaussian filter. If zero, 
                    then no filter was applied. Must be odd.
                std_dev: Standard deviation for the Gaussian kernel. If
                    std_dev < 0, then the standard deviation is
                    automatically calucated using, 
                    0.3*((k_size - 1)*0.5 - 1) + 0.8
            
            ["median", k_size]
                k_size: Kernel size for the median filter. If zero, 
                    then no filter was applied. Must be odd.
            
            ["bilateral", d_size, sig_intsty]
                d_size: Diameter of each pixel neighborhood (must be 
                    even). If zero, then no filter was applied.
                sig_intsty: Filter sigma in the color space. A larger
                    value means that farther colors within d_size will 
                    be mixed together, resulting in larger areas of
                    semi-equal color. 
    
    quiet_in: A boolean that determiens if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing.      

    ---- RETURNED ---- 
    [img_fltr]: Returns the resultant image after performing the
        image processing procedures. img_fltr is in the same format as
        img_in.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output. 
    """

    # ---- Start Local Copies ----
    img = img_in.copy()
    fltr_params = fltr_params_in
    quiet = quiet_in
    # ---- End Local Copies ----

    fltr_name = (fltr_params[0]).lower()

    # Extract the filter parameters depending on the filter type, and
    # then apply said filter. Note, it is possible that some parameters,
    # like kernel size, are zero which would throw an error if used in
    # some of OpenCV's filters. If kernel size is zero, then that means
    # just show the original image -- no changes were made.
    if fltr_name == "average": # Equal-weighted average kernel
        avg_ksize = fltr_params[1]
        if avg_ksize == (0, 0): # No filtering performed, keep original
            return img 
        else: # Apply filter
            img_fltr = cv.blur(img, avg_ksize)

        if not quiet:
             print(f"\nSuccessfully appled the 'average' blur filter:\n"\
                f"    Kernel Size = {(avg_ksize, avg_ksize)}")

    elif fltr_name == "gaussian": # Gaussian-weighted kernel
        gaus_ksize = fltr_params[1]
        gaus_sdev = fltr_params[2]
        if gaus_ksize == (0, 0): # No filtering performed, keep original
            return img 
        else: # Apply filter
            if gaus_sdev < 0: # Calculate automatically if less than 0
                gaus_sdev = 0.3*((gaus_ksize - 1)*0.5 - 1) + 0.8

            if gaus_ksize[0] % 2 == 0: # Must be odd for this filter
                gaus_ksize[0] += -1
                gaus_ksize[1] += -1

            img_fltr = cv.GaussianBlur(img, gaus_ksize, gaus_sdev)

        if not quiet:
            print(f"\nSuccessfully applied the 'gaussian' blur filter:\n"\
                f"    Kernel Size = {(gaus_ksize, gaus_ksize)}\n"\
                f"    Standard Deviation = {gaus_sdev}") 

    elif fltr_name == "median": # Center weight of kernal is median value
        med_ksize = fltr_params[1]
        if med_ksize == 0: # No filtering performed, keep original
            return img 
        else: # Apply filter
            if med_ksize % 2 == 0: # Must be odd for this filter
                med_ksize += -1
            img_fltr = cv.medianBlur(img, med_ksize) 

        if not quiet:
            print(f"\nSuccessfully applied the 'median' blur filter:\n"\
                f"    Kernel Size = {med_ksize}")

    elif fltr_name == "bilateral": # Edge-preserving Gaussian kernel
        bil_dsize = fltr_params[1]
        bil_sint = fltr_params[2]
        if bil_dsize == 0: # No filtering performed, keep original
            return img 
        else: # Apply filter
            if bil_dsize % 2 != 0: # Should be even for this filter
                bil_dsize += 1
            img_fltr = cv.bilateralFilter(img, bil_dsize, bil_sint, bil_sint)

        if not quiet:
            print(f"\nSuccessfully applied the 'bilateral' blur filter:\n"\
                f"    Pixel Neighborhood = {bil_dsize}\n" \
                f"    Intensity Threshold = {bil_sint}")

    else:
        print(f"\nERROR: {fltr_name} is not a supported blur filter.")
        img_fltr = img

    return img_fltr


def interact_driver_sharpen(img_in, fltr_name_in):
    """
    Interactively implements a sharpen filter on the provided image. The
    parameters for the image processing are controlled by trackbars in
    the interactive window, and the resultant effects are updated in
    real-time. Upon hitting the Enter or Esc keys, the interactive 
    window closes, and then this function returns the final image. These
    image processing operations can also be called edge enhancements.
    This function is related to apply_driver_sharpen(...).

    ---- INPUT ARGUMENTS ----
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel. 

    fltr_name_in: A string that represents the name of the sharpen
        filter to be applied. Can be either "unsharp", "laplacian", or
        "canny", 
            
            "unsharp": Applies a conventional unsharp mask based on 
                either a Gaussian or median kernel. See digital unsharp 
                masking at, "https://en.wikipedia.org/wiki/
                Unsharp_masking"
            
            "laplacian": Uses the Laplacian (i.e., second derivatives)
                to enhance the image's edges.
            
            "canny": Uses the Canny edge detection algorithm to add
                contrast to the edges. These enhancements may be 
                somewhat discrete, rather than smooth gradients.

    ---- RETURNED ---- 
    [img_sharp]: Returns the final image after closing the interactive
        session. img_sharp is in the same format as img_in.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """

    # ---- Start Local Copies ----
    fltr_name = fltr_name_in.lower() # Ensure string is all lowercase
    # ---- End Local Copies ----

    # List of strings of the supported filter types
    fltr_list = ["unsharp", "laplacian", "canny"]

    # Ensure that the filter name is recognized
    if fltr_name not in fltr_list:
        print(f"\n{fltr_name_in} is not a supported filter type. Supported "\
            f"filter types are: {fltr_list}\nDefaulting to an Unsharp Mask.")
        fltr_name = "unsharp"

    # -------- Filter Type: "unsharp" --------
    if fltr_name == "unsharp":
        [img_sharp, sharp_params] = ifun.interact_unsharp_mask(img_in)

    # -------- Filter Type: "laplacian" --------
    elif fltr_name == "laplacian":
        [img_sharp, sharp_params] = ifun.interact_laplacian_sharp(img_in)

    # -------- Filter Type: "canny" --------
    elif fltr_name == "canny":
        [img_sharp, sharp_params] = ifun.interact_canny_sharp(img_in)

    # Using this function just to write to standard output
    apply_driver_sharpen(img_in, sharp_params)

    return img_sharp


def apply_driver_sharpen(img_in, fltr_params_in, quiet_in=False):
    """
    Applies a sharpen filter, much like interact_driver_sharpen(...),
    but in a non-interactive way. Note, sharpening is effectively the 
    same as edge-enhancement.

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel. 

    [fltr_params_in]: A list of parameters needed to perform the sharpen
        operation. The first parameter is a string, which determines
        what type of sharpen filter to be applied, as well as the 
        definitions of the remaining parameters. fltr_params_in[0] can
        either be "unsharp", "laplacian", "canny. Example parameter 
        lists are given below for each type,
            
            ["unsharp", cur_amount, k_size, fltr_type, std_dev]
                cur_amount: A percentage value of how much to blend
                    the sharpening effect onto the original value. It
                    corresponds to (amount)*100 in the equation above.
                    If zero, then no filter is applied.
                k_size: Kernel size for the blur filter. If zero, 
                    then no filter was applied. Must be odd.
                fltr_type: Either 0 or 1. If 0, then the blurred image
                    used in the procedure utilizes the Gaussian filter.
                    If 1, then the median filter is used.
                std_dev: Corresponds to the standard deviation used in
                    defining the Gaussian kernel for the Gaussian
                    filter. This parameter is ignored if the median
                    filter is used. If std_dev < 0, then the standard
                    deviation is automatically calculated.
            
            ["laplacian", cur_amount, blur_k_size, lap_k_size,  
            fltr_type, blur_std_dev]
                cur_amount: A percentage value of how much to blend
                    the sharpening effect onto the original value. It
                    corresponds to (amount)*100 in the equation above.
                    If zero, then no filter is applied.
                blur_k_size: Kernel size for the blur filter that is
                    applied prior to using the Laplacian kernel. If 
                    zero, then no blur filter is applied. Must be odd.
                lap_k_size: Kernel size for the Laplacian kernel used
                    to find the edges of the original image.
                fltr_type: Either 0 or 1. If 0, then the blurred image
                    used in the procedure utilizes the Gaussian filter.
                    If 1, then the median filter is used.
                blur_std_dev: Corresponds to the standard deviation used
                    in defining the Gaussian kernel for the Gaussian
                    blur. This parameter is ignored if the median blur
                    is used. If std_dev < 0, then the standard deviation
                    is automatically calculated.
            
            ["canny", k_size, cur_amount, thresh1, thresh2]
                k_size: Kernel size for the Canny edge algorithm. If
                    zero, then no filter was applied. Must be odd.
                cur_amount: A percentage value of how much to blend
                    the sharpening effect onto the original value. It
                    corresponds to (amount)*100 in the equation above.
                    If zero, then no filter is applied.
                thresh1: Related to the hystersis thresholding algorithm
                    in the Canny Edge Detection procedure. Related to 
                    minimum threshold used to determine what an edge is.
                thresh2: Related to the hystersis thresholding algorithm
                    in the Canny Edge Detection procedure. Related to 
                    maximum threshold used to determine what an edge is.
    
    quiet_in: A boolean that determiens if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing. 

    ---- RETURNED ---- 
    [img_sharp]: Returns the resultant image after performing the
        image processing procedures. img_sharp is in the same format as
        img_in.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """

    # ---- Start Local Copies ----
    img = img_in.copy()
    fltr_params = fltr_params_in
    quiet = quiet_in
    # ---- End Local Copies ----

    fltr_name = (fltr_params[0]).lower()

    # -------- Filter Type: "unsharp" --------
    if fltr_name == "unsharp":
        fltr_params_2 = fltr_params[1:]

        if fltr_params_2[1] % 2 == 0: # Must be odd for the blur filters
            fltr_params_2[1] += -1

        if fltr_params_2[0] <= 0:
            return img
        elif fltr_params_2[1] < 1:
            return img

        img_sharp = wrap.unsharp_mask(img, fltr_params_2)

        if not quiet:
            print(f"\nSuccessfully applied the 'unsharp' mask:\n"\
                f"    Amount: {fltr_params_2[0]}%\n"\
                f"    Radius of Blur Kernel: {fltr_params_2[1]}")

            if fltr_params_2[2]:
                print("    Blur Filter Type: Median")
            else:
                print("    Blur Filter Type: Gaussian\n"\
                    f"    Standard Deviation: {fltr_params_2[3]}")

    # -------- Filter Type: "laplacian" --------
    elif fltr_name == "laplacian":
        fltr_params_2 = fltr_params[1:]

        if fltr_params_2[1] % 2 == 0: # Must be odd for the Gaussian blur
            fltr_params_2[1] += -1

        if fltr_params_2[2] % 2 == 0: 
            fltr_params_2[2] += -1

        if fltr_params_2[0] <= 0:
            return img
        elif (fltr_params_2[2] < 1) or (fltr_params_2[1] < 1):
            return img

        img_sharp = wrap.laplacian_sharp(img, fltr_params_2)

        if not quiet:
            print(f"\nSuccessfully applied the 'laplacian' mask:\n"\
                f"    Amount: {fltr_params_2[0]}%\n"\
                f"    Radius of Blur Kernel: {fltr_params_2[1]}\n"\
                f"    Radius of Laplacian Kernel: {fltr_params_2[2]}")

            if fltr_params_2[3]:
                print("    Blur Filter Type: Median")
            else:
                print("    Blur Filter Type: Gaussian\n"\
                    f"    Standard Deviation: {fltr_params_2[4]}")

    # -------- Filter Type: "canny" --------
    elif fltr_name == "canny":
        fltr_params_2 = fltr_params[1:]

        if fltr_params_2[0] % 2 == 0: 
            fltr_params_2[0] += -1

        img_sharp = wrap.canny_sharp(img, fltr_params_2)

        if not quiet:
            print(f"\nSuccessfully applied the 'canny' edge mask:\n"\
                f"    Radius of Canny Edge Kernel: {fltr_params_2[0]}\n"\
                f"    Amount: {fltr_params_2[1]}%\n"\
                f"    Hysteresis Threshold 1: {fltr_params_2[2]}\n"\
                f"    Hysteresis Threshold 2: {fltr_params_2[3]}")

    else:
        print(f"\nERROR: {fltr_name} is not a supported sharpen filter.")
        img_sharp = img
        
    return img_sharp


def interact_driver_equalize(img_in, eq_name_in):
    """
    Interactively equalizes the histogram on the provided image. The
    parameters for the image processing are controlled by trackbars in
    the interactive window, and the resultant effects are updated in
    real-time. Upon hitting the Enter or Esc keys, the interactive 
    window closes, and then this function returns the final image. These
    image processing operations can also be called contrast enhancements.
    This function is related to apply_driver_equalize(...). More details
    about equalization can be found at, "https://docs.opencv.org/4.2.0/
    d5/daf/tutorial_py_histogram_equalization.html"

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel. 
    
    eq_name_in: A string that represents the type of equalization 
        technique to use. Can be either "global" or "adaptive".
            
            "global": Warning, this is not interactive! Values are
                automatically calculated and applied to the entire 
                image.
            
            "adaptive": Locally adaptive procedure to enhance the
                contrast. This particular process is known as Contrast
                Limited Adaptive Histogram Equalization (CLAHE).

    ---- RETURNED ---- 
    [img_eq]: Returns the final image after closing the interactive
        session. img_eq is in the same format as img_in.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """

    # ---- Start Local Copies ----
    eq_name = eq_name_in.lower() # Ensure string is all lowercase
    # ---- End Local Copies ----

    # List of strings of the supported filter types
    eq_list = ["global", "adaptive"]

    # Ensure that the filter name is recognized
    if eq_name not in eq_list:
        print(f"\n{eq_name_in} is not a supported filter type. Supported "\
            f"filter types are: {eq_list}\nDefaulting to a Adaptive.")
        eq_name = "adaptive"

    # -------- Filter Type: "global" --------
    if eq_name == "global":
        img_eq = wrap.global_equalize(img_in)
        eq_params = ["global"]

    # -------- Filter Type: "adaptive" --------
    elif eq_name == "adaptive":
        [img_eq, eq_params] = ifun.interact_adaptive_equalize(img_in)

    # Using this function just to write to standard output
    apply_driver_equalize(img_in, eq_params)

    return img_eq


def apply_driver_equalize(img_in, eq_params_in, quiet_in=False):
    """
    Equalizes the image's intensity histogram, much like 
    interact_driver_equalize(...), but in a non-interactive way. Note,
    equalizing the histogram is a method to enhance contrast. More 
    details about equalization can be found at, "https://docs.opencv.org
    /4.2.0/d5/daf/tutorial_py_histogram_equalization.html"

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.
    
    [eq_params_in]: A list of parameters needed to perform the 
        equalization operation. The first parameter is a string, which
        determines what type of equalization to be applied, as well as
        the  definitions of the remaining parameters. eq_params_in[0]
        can either be "global" or "adaptive". Example parameter lists
        are  given below for each type,
            
            ["global"]
                No additional parameters required
            
            ["adaptive", clip_limit, grid_size]
                clip_limit: Threshold for contrast limiting
                grid_size: Size of grid for histogram equalization. 
                    Input image will be divided into equally sized 
                    rectangular tiles. tileGridSize defines the number 
                    of tiles in row and column. If zero, the original
                    image is returned.]
    
    quiet_in: A boolean that determiens if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing. 

    ---- RETURNED ---- 
    [img_eq]: Returns the resultant image after performing the
        image processing procedures. img_eq is in the same format as
        img_in.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """

    # ---- Start Local Copies ----
    img = img_in.copy()
    eq_params = eq_params_in
    quiet = quiet_in
    # ---- End Local Copies ----

    eq_name = (eq_params[0]).lower()

    # -------- Filter Type: "global" --------
    if eq_name == "global":
        img_eq = wrap.global_equalize(img)
        if not quiet:
            print("\nSuccessfully applied 'global' equalization")

    # -------- Filter Type: "adaptive" --------
    elif eq_name == "adaptive":
        eq_params_2 = eq_params[1:]

        if eq_params_2[1] < 1:
            return img

        img_eq = wrap.adaptive_equalize(img, eq_params_2)

        if not quiet:
            print("\nSuccessfully applied locally 'adaptive' equalization:\n"\
                f"    Clip Limit: {eq_params_2[0]}\n"\
                f"    Tile Size: ({eq_params_2[1]},{eq_params_2[1]})")

    else:
        print(f"\nERROR: {eq_name} is not a supported equalization "\
            "operation.")
        img_eq = img

    return img_eq


def interact_driver_morph(img_in):
    """
    Creates an interactive session to perform various types of 
    morphological transformations (i.e., dilation and erosion). The user
    can either perform just dilation operations, or just erosion
    transformations. Alternatively, the user can perform either an open
    transformation or a close transformation. Note, an open
    transformation is an erosion followed by a dilation, and it can be
    helpful in removing noise. Conversely, a close transformation is a
    dilation followed by an erosion, and it can be useful in closing
    small holes. The user can update the various parameters and see the
    effects in real-time within the interactive window. Upon hitting the
    Enter or Esc keys, the interactive window closes, and then this 
    function returns the final image. This function is closely related
    to apply_driver_morph(...).

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel. 

    ---- RETURNED ---- 
    [img_morph]: Returns the final image after closing the interactive
        session. img_morph is in the same format as img_in.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """

    [img_morph, morph_params] = ifun.interact_morph(img_in)

    # Using this function just to write to standard output
    apply_driver_morph(img_in, morph_params)

    return img_morph


def apply_driver_morph(img_in, morph_params_in, quiet_in=False):
    """
    Applies a morphological operation (i.e., dilation and erosion
    transformations), much like interact_driver_morph(...) but not in an
    interactive way. The user can either perform just dilation
    operations, or just erosion transformations. Alternatively, the user
    can perform either an open transformation or a close transformation.
    Note, an open transformation is an erosion followed by a dilation,
    and it can be helpful in removing noise. Conversely, a close
    transformation is a dilation followed by an erosion, and it can be
    useful in closing small holes.

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel. 

    [morph_params_in]: A list parameters needed to perform the desired
        morphological transformations. A more detailed explanation of
        each parameter in morph_params_in is given below,
            
            [flag_open_close, flag_rect_ellps, k_size, num_erode,
            num_dilate]
                flag_open_close: Either 0 or 1, and determines if the
                    erosion operations should occur prior to dilation
                    operations, or the other way around. If 0, then an 
                    open operation is done, which is erosion followed by 
                    dilation (removes noise). If 1, then a close 
                    operation is done, which is a dilation followed by
                    erosion (closes small holes). 
                flag_rect_ellps: Either 0 or 1. If 0, then a rectangular
                    kernel is used for the morphological operations. If
                    1, then an elliptical kernel is used.
                k_size: Kernel size for the morphological operations. If
                    less than 1, then the original image is returned.
                num_erode: Number of erosion operations to perform
                num_dilate: Number of dilation operations to perform
    
    quiet_in: A boolean that determines if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing. 

    ---- RETURNED ---- 
    [img_morph]: Returns the final image after the morphological 
        transformations. img_morph is in the same format as img_in.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """

    # ---- Start Local Copies ----
    morph_params = morph_params_in
    quiet = quiet_in

    # 0 is open => erosion followed by dilation (removes noise)
    # 1 is close => dilation followed by erosion (closes small holes)
    flag_open_close = morph_params[0]

    # 0 is rectangular kernel, 1 is elliptical kernel
    flag_rect_ellps = morph_params[1]

    # Width (or diameter) of the kernel in pixels
    k_size = morph_params[2]

    # Number of iterations of the erode and dilation operations
    num_erode = morph_params[3]
    num_dilate = morph_params[4]

    if k_size < 1: # Nothing to report if kept original image
        return img_in
    elif (num_dilate < 1) and (num_erode < 1):
        return img_in

    # Perform successive morphological operations
    img_morph = wrap.multi_morph(img_in, morph_params)

    # Write out comments to standard output based on what was done
    if flag_open_close: # Close
        if (num_dilate >= 1) and (num_erode >= 1):
            if not quiet:
                print("\nSuccessfully applied morphological operations:")
                if flag_rect_ellps:
                    print("    Kernel Shape: Ellipse\n"\
                         f"    Kernel Diameter: {k_size}")
                else:
                    print("    Kernel Shape: Rectangle\n"\
                         f"    Kernel Width: {k_size}")

                op_hist = f"{num_dilate} X Dilate   followed by   "\
                    f"{num_erode} X Erode"
                print("    Kernel Operations: " + op_hist)

        elif num_dilate < 1: # Only do a erosion
            if not quiet:
                print("\nSuccessfully applied morphological operations:")
                if flag_rect_ellps:
                    print("    Kernel Shape: Ellipse\n"\
                         f"    Kernel Diameter: {k_size}")
                else:
                    print("    Kernel Shape: Rectangle\n"\
                         f"    Kernel Width: {k_size}")

                op_hist = f"0 X Dilate   followed by   "\
                    f"{num_erode} X Erode"
                print("    Kernel Operations: " + op_hist)

        else: # Only do an dilation
            if not quiet:
                print("\nSuccessfully applied morphological operations:")
                if flag_rect_ellps:
                    print("    Kernel Shape: Ellipse\n"\
                         f"    Kernel Diameter: {k_size}")
                else:
                    print("    Kernel Shape: Rectangle\n"\
                         f"    Kernel Width: {k_size}")

                op_hist = f"{num_dilate} X Dilate   followed by   "\
                    f"0 X Erode"
                print("    Kernel Operations: " + op_hist)

    else: # Open
        if (num_dilate >= 1) and (num_erode >= 1):
            if not quiet:
                print("\nSuccessfully applied morphological operations:")
                if flag_rect_ellps:
                    print("    Kernel Shape: Ellipse\n"\
                         f"    Kernel Diameter: {k_size}")
                else:
                    print("    Kernel Shape: Rectangle\n"\
                         f"    Kernel Width: {k_size}")

                op_hist = f"{num_erode} X Erode   followed by   "\
                    f"{num_dilate} X Dilate"
                print("    Kernel Operations: " + op_hist)

        elif num_dilate < 1: # Only do a erosion
            if not quiet:
                print("\nSuccessfully applied morphological operations:")
                if flag_rect_ellps:
                    print("    Kernel Shape: Ellipse\n"\
                         f"    Kernel Diameter: {k_size}")
                else:
                    print("    Kernel Shape: Rectangle\n"\
                         f"    Kernel Width: {k_size}")

                op_hist = f"{num_erode} X Erode   followed by   "\
                    f"0 X Dilate"
                print("    Kernel Operations: " + op_hist)

        else: # Only do an dilation
            if not quiet:
                print("\nSuccessfully applied morphological operations:")
                if flag_rect_ellps:
                    print("    Kernel Shape: Ellipse\n"\
                         f"    Kernel Diameter: {k_size}")
                else:
                    print("    Kernel Shape: Rectangle\n"\
                         f"    Kernel Width: {k_size}")

                op_hist = f"0 X Erode   followed by   "\
                    f"{num_dilate} X Dilate"
                print("    Kernel Operations: " + op_hist)

    return img_morph


def interact_driver_thresh(img_in, thsh_name_in):
    """
    Creates an interactive session to perform a black and white
    threshold of a grayscale image. This is also known as binarization.
    Both global and locally adaptive methods are supported. The final
    image will be composed of just black and white intensities, which
    correspond to 0 and 255, respectively. Upon hitting the Enter or Esc
    keys, the interactive window closes, and then this  function returns
    the final image. This function is closely related to
    apply_driver_thresh(...).

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel. 
    
    thsh_name_in: A string that represents the name of the type of
        thresholding operation to be performed. Supported methods are
        "global", "adaptive_mean", and "adaptive_gaussian". Basic 
        descriptions are given below, but for more details, visit
        "https://docs.opencv.org/4.2.0/d7/d4d/tutorial_py_
        thresholding.html"

            "global": Uses the same threshold value (between 0 and 255)
                for every pixel. The threshold can be set manually or 
                automatically using Otsu's method.

            "adaptive_mean": A locally adaptive method. The threshold
                value is the mean of the neighborhood area minus a
                constant.

            "adaptive_gaussian": A locally adaptive method. The threshold
                value is a gaussian-weighted sum of the neighborhood area
                minus a constant.

    ---- RETURNED ---- 
    [img_thsh]: Returns the final image after closing the interactive
        session. img_thsh is in the same format as img_in.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """

    # ---- Start Local Copies ----
    thsh_name = thsh_name_in.lower()
    # ---- End Local Copies ----

    # List of strings of the supported filter types
    thsh_list = ["global", "adaptive_mean", "adaptive_gaussian"]

    # Ensure that the filter name is recognized
    if thsh_name not in thsh_list:
        print(f"\n{thsh_name} is not a supported threshold type. Supported "\
            f"threshold types are: {thsh_list}\nDefaulting to "\
            "adaptive_gaussian.")
        thsh_name = "adaptive_gaussian"

    if thsh_name == "global":
        [img_thsh, thsh_params] = ifun.interact_global_threshold(img_in)

    elif thsh_name == "adaptive_mean":
        [img_thsh, thsh_params] = ifun.interact_mean_threshold(img_in)

    elif thsh_name == "adaptive_gaussian":
        [img_thsh, thsh_params] = ifun.interact_gaussian_threshold(img_in)

    # Using this function just to write to standard output
    apply_driver_thresh(img_in, thsh_params)

    return img_thsh


def apply_driver_thresh(img_in, thsh_params_in, quiet_in=False):
    """
    Perform a black and white threshold of a grayscale image. This is
    also known as binarization. Both global and locally adaptive methods
    are supported. The final image will be composed of just black and
    white intensities, which correspond to 0 and 255, respectively. This
    function is similar to interact_driver_thresh(...), but is
    non-interactive.

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel. 

    [thsh_params_in]: A list of parameters to perform the desired 
        threshold operation. The first parameter is a string, which
        determines what type of thresholding is performed, as well as
        the definitions of the remaining parameters. thsh_params_in[0]
        can be either "global", "adaptive_mean", or "adaptive_gaussian".
        Example parameter lists are given below for each type,

            ["global", cur_thsh]
                cur_thsh: The specified intensity to use as the 
                    threshold limit between black and white. It should
                    be an integer between 0 and 255. However, if
                    cur_thsh < 0, then cur_thsh will be calculated
                    automatically using Otsu's method.

            ["adaptive_mean", blk_size, c_offset]
                blk_size: Size of a pixel neighborhood that is used to
                    calculate a threshold value for the current pixel.
                    Must be odd, and if zero, no thresholding is done.
                c_offset: A constant offset value applied to the mean
                    or weighted mean of the intensities

            ["adaptive_gaussian", blk_size, c_offset]
                blk_size: Size of a pixel neighborhood that is used to
                    calculate a threshold value for the current pixel.
                    Must be odd, and if zero, no thresholding is done.
                c_offset: A constant offset value applied to the mean
                    or weighted mean of the intensities

    quiet_in: A boolean that determiens if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing

    ---- RETURNED ---- 
    [img_thsh]: Returns the final image after thresholding. img_thsh is
        in the same format as img_in.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """

    # ---- Start Local Copies ----
    img = img_in.copy()
    thsh_params = thsh_params_in
    quiet = quiet_in
    # ---- End Local Copies ----

    thsh_name = (thsh_params[0]).lower()

    # -------- Filter Type: "global" --------
    if thsh_name == "global":
        thsh_params_2 = thsh_params[1:]

        # If the threshold value is < 0, apply automatic Otsu method
        if thsh_params_2[0] < 0:
            [ret_thsh, img_thsh] = cv.threshold(img, 0, 255, 
                cv.THRESH_BINARY+cv.THRESH_OTSU)

            if not quiet:
                print(f"\nSuccessfully applied 'global' thresholding:\n"\
                    f"    Threshold Value (Otsu Method): {ret_thsh}") 

        else:
            if thsh_params_2[0] > 255:
                thsh_params_2[0] = 255

            [ret_thsh, img_thsh] = cv.threshold(img, thsh_params_2[0], 255, 
                cv.THRESH_BINARY)

            if not quiet:
                print(f"\nSuccessfully applied 'global' thresholding:\n"\
                    f"    Threshold Value: {thsh_params_2[0]}")         

        return img_thsh


    # -------- Filter Type: "adaptive_mean" --------
    elif thsh_name == "adaptive_mean":
        thsh_params_2 = thsh_params[1:]

        if thsh_params_2[0] <= 0: # Return original image
            img_thsh = img

        else:
            if thsh_params_2[0] % 2 == 0: # Must be odd (but not 1) and > 0
                thsh_params_2[0] += -1

            if thsh_params_2[0] < 3:
                thsh_params_2[0] = 3

            img_thsh = cv.adaptiveThreshold(img, 255, 
                cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 
                thsh_params_2[0], thsh_params_2[1])

            if not quiet:
                print(f"\nSuccessfully applied 'adaptive mean' threshold:\n"\
                    f"    Block Size: {thsh_params_2[0]} pixels\n"\
                    f"    Intensity Offset: {thsh_params_2[1]}")

        return img_thsh

    # -------- Filter Type: "adaptive_gaussian" --------
    elif thsh_name == "adaptive_gaussian":
        thsh_params_2 = thsh_params[1:]

        if thsh_params_2[0] <= 0: # Return original image
            img_thsh = img

        else:
            if thsh_params_2[0] % 2 == 0: # Must be odd (but not 1) and > 0
                thsh_params_2[0] += -1

            if thsh_params_2[0] < 3:
                thsh_params_2[0] = 3

            img_thsh = cv.adaptiveThreshold(img, 255, 
                cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 
                thsh_params_2[0], thsh_params_2[1])

            if not quiet:
                print(f"\nSuccessfully applied 'adaptive gaussian' "\
                    f"threshold:\n"\
                    f"    Block Size: {thsh_params_2[0]} pixels\n"\
                    f"    Intensity Offset: {thsh_params_2[1]}")

        return img_thsh

    else:
        print(f"\nERROR: {thsh_name} is not a supported threshold "\
            "operation.")


def interact_driver_blob_fill(img_in):
    """
    Removes blobs (i.e., islands) of pixels from a binarized image in an
    interactive sesssion. Blobs are identified based on user-selected
    thresholding parameters related to the blob size (in pixels),
    circularity, and aspect ratio. It is assumed that the blobs to be
    removed are white, and thus, will be filled in with black pixels to
    effectively delete them. Upon hitting Enter during the interactive
    session, the interactive window closes and the final image is
    returned. This function is related to apply_driver_blob_fill(...).

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel. 

    ---- RETURNED ---- 
    [img_blob]: Returns the final image after closing the interactive
        session. img_blob is in the same format as img_in.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """
    [img_blob, blob_params] = ifun.interact_blob_fill(img_in)

    # Using this function just to write to standard output
    apply_driver_blob_fill(img_in, blob_params)

    return img_blob


def apply_driver_blob_fill(img_in, blob_params_in, quiet_in=False):
    """
    Removes blobs (i.e., islands) of pixels from a binarized image like
    interact_driver_blob_fill(...), but as a batch process rather than in
    an interactive session. Blobs are identified based on user-selected
    thresholding parameters related to the blob size (in pixels),
    circularity, and aspect ratio. 

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel. 

    [blob_params_in]: A list of threshold parameters necessary to 
        identify blobs which should get filled in with a specified 
        color. More details on the parameters of blob_params_in is given
        below,

        [area_thresh_min, area_thresh_max, circty_thresh_min,
        circty_thresh_max, ar_min, ar_max, COLOR]
                
            area_thresh_min: Lower threshold bound for identifying blobs
                to be deleted. Related to the blob size based on the
                total number of pixels.
                
            area_thresh_max: Upper threshold bound for identifying blobs
                to be deleted. Related to the blob size based on the
                total number of pixels.
                
            circty_thresh_min: Lower threshold bound for identifying
                blobs to be deleted. Related to the circularity of the
                blob, based on, 4*pi*area/(perimeter)^2
                
            circty_thresh_max: Upper threshold bound for identifying
                blobs to be deleted. Related to the circularity of the
                blob, based on, 4*pi*area/(perimeter)^2
                
            ar_min: Lower threshold bound for identifying blobs to  be
                deleted. Related to the aspect ratio, which is the width
                over length of the bounding rectangle.
                
            ar_max: Upper threshold bound for identifying blobs to  be
                deleted. Related to the aspect ratio, which is the width
                over length of the bounding rectangle.
                
            COLOR: The color used to fill in the pixels of each blob.
                Should be a tuple containing three integers, each
                corresponding to the intensity of the color channels (in
                BGR format). For example, (0, 0, 0) corresponds to
                black.

    quiet_in: A boolean that determiens if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing

    ---- RETURNED ---- 
    [img_blob]: Returns the final image after filling-in the identified
        blobs. img_blob is in the same format as img_in.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """
    
    img = img_in.copy()
    blob_params = blob_params_in
    quiet = quiet_in

    area_thresh_min = blob_params[0]
    area_thresh_max = blob_params[1]
    circty_thresh_min = blob_params[2]
    circty_thresh_max = blob_params[3]
    ar_min = blob_params[4]
    ar_max = blob_params[5]
    blob_color = blob_params[6]

    PI = float(3.14159265358979323846264338327950288419716939937510)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    area_thresh_min = int(round(area_thresh_min))
    area_thresh_max = int(round(area_thresh_max))

    if area_thresh_min > area_thresh_max:
        area_thresh_min = area_thresh_max
    elif area_thresh_max < area_thresh_min:
        area_thresh_max = area_thresh_min

    if circty_thresh_min > circty_thresh_max:
        circty_thresh_min = circty_thresh_max
    elif circty_thresh_max < circty_thresh_min:
        circty_thresh_max = circty_thresh_min

    if ar_min > ar_max:
        ar_min = ar_max
    elif ar_max < ar_min:
        ar_max = ar_min

    [img_contrs, contr_hierarchy] = cv.findContours(img, cv.RETR_LIST, 
        cv.CHAIN_APPROX_SIMPLE)

    if img_contrs:

        contrs_area = []
        contrs_perim = []
        contrs_circty = []
        contrs_del = []

        for contr in img_contrs:
            cur_area = cv.contourArea(contr)
            contrs_area.append(cur_area)

            cur_perim = cv.arcLength(contr, True)
            contrs_perim.append(cur_perim)

            if cur_perim != 0:
                cur_circty = 4.0*PI*cur_area/(cur_perim*cur_perim)
            else:
                cur_circty = 0.0

            # Could also get the aspect ratio and compare against that
            [x_rect, y_rect, w_rect, h_rect] = cv.boundingRect(contr)
            cur_ar = float(w_rect)/h_rect 

            if (area_thresh_min <= cur_area <= area_thresh_max) and\
                (circty_thresh_min <= cur_circty <= circty_thresh_max) and\
                (ar_min <= cur_ar <= ar_max):
             
                contrs_del.append(contr)        

    # Fill the white blobs with black pixels, effectively deleting them
    if contrs_del: 
        for contr in contrs_del:
            cv.drawContours(img, [contr], 0, blob_color, thickness=cv.FILLED, 
                lineType=cv.LINE_8)

    num_del_blob = len(contrs_del)

    if not quiet:
        print(f"\nSuccessfully filled in {num_del_blob} blobs using the "\
            "following criterion:\n"\
            f"    Min. Area Threshold: {area_thresh_min} pixels\n"\
            f"    Max. Area Threshold: {area_thresh_max} pixels\n"\
            f"    Min. Circularity: {circty_thresh_min}\n"\
            f"    Max. Circularity: {circty_thresh_max}\n"\
            f"    Min. Aspect Ratio: {ar_min}\n"\
            f"    Max. Aspect Ratio: {ar_max}")

    return img


def interact_driver_denoise(img_in):
    """
    Create an interactive session that changes the parameters necessary
    for the OpenCV's advanced non-local means denoising algorithm. The
    user can change the procedure's parameters and get updates in real-
    time. However, this denoising procedure is very computationally
    expensive, so the window may stall for short periods of time with
    large patch sizes. Compared to blur filters, like Gaussian blur,
    this procedure usually does a better job at removing noise while
    maintaining the integrity of the original signal. So, in a sense,
    this procedure is edge preserving, or at least, edge friendly. Upon
    hitting Enter, the interactive window closes, and the final image is
    returned. This function is related to apply_driver_denoise(...).

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel. 

    ---- RETURNED ---- 
    [img_denoise]: Returns the final image after closing the interactive
        session. img_denoise is in the same format as img_in

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """

    [img_denoise, denoise_params] = ifun.interact_denoise(img_in)

    # Using this function just to write to standard output
    apply_driver_denoise(img_in, denoise_params)

    return img_denoise


def apply_driver_denoise(img_in, denoise_params_in, quiet_in=False):
    """
    Implements OpenCV's advanced non-local means denoising algorithm
    like interact_driver_denoise(...), but as a batch process rather 
    than being interactive. This procedure is rather computationally 
    intensive, particularly for large patch sizes. However, Compared to
    blur filters, like Gaussian blur, this procedure usually does a
    better job at removing noise while maintaining the integrity of the
    original signal. So, in a sense, this procedure is edge preserving,
    or at least, edge friendly.

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel. 

    [denoise_params_in]: A list of parameters for input to the non-local
        means denoising algorithm. More details on the parameters of
        denoise_params_in is given below,
    
            [cur_h, cur_tsize, cur_wsize]
                
                cur_h: Parameter regulating filter strength. Big cur_h
                    value perfectly removes noise but also removes image
                    details, smaller cur_h value preserves details but 
                    also preserves some noise.
                
                cur_tsize: Size in pixels of the template patch that is
                    used to compute weights. Should be odd.
                
                cur_wsize: Size in pixels of the window that is used to
                    compute weighted average for given pixel. Should be
                    odd. Affects performance linearly.

    quiet_in: A boolean that determiens if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing

    ---- RETURNED ---- 
    [img_denoise]: Returns the final image after applying the non-local
        means denoising algorithm. img_denoise is in the same format as
        img_in

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """

    img = img_in.copy()
    denoise_params = denoise_params_in
    quiet = quiet_in

    cur_h = denoise_params[0]
    cur_tsize = denoise_params[1]
    cur_wsize = denoise_params[2]

    if cur_h <= 0:
        img_denoise = img

    else:
        # Must be odd for the denoise function
        if cur_tsize % 2 == 0: 
            cur_tsize += -1

        if cur_wsize % 2 == 0: 
            cur_wsize += -1

        img_denoise = cv.fastNlMeansDenoising(img, h=cur_h, 
            templateWindowSize=cur_tsize, searchWindowSize=cur_wsize)

        if not quiet:
            print("\nSuccessfully applied the non-local means denoising"\
                "filter:\n"\
                f"    Filter Strength: {cur_h}\n"\
                f"    Template Patch Size: {cur_tsize}\n"
                f"    Current Window Size: {cur_wsize}")

    return img_denoise
