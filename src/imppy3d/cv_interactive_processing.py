# Import external dependencies
import numpy as np
import cv2 as cv

# Import local packages
from . import cv_processing_wrappers as wrap
from . import cv_driver_functions as drv


def do_nothing(x):
    """
    Literally does nothing. Need a dummy function for 
    cv.createTrackbar(...), which is used in later functions.
    """
    pass


def interact_average_filter(img_in):
    """
    Applies the averaging blur filter in an interactive way. The user
    can update the filter's parameters and see the updates in real-time.
    Upon hitting Enter, the interactive window closes, the final image
    is returned, along with the necessary filter parameters to recreate
    the image. These returned parameters work as input to the related
    driver function, apply_driver_blur(...).

     ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

     ---- RETURNED ---- 
    [[img_avg], [fltr_params]]: Returns a list containing the resultant 
        image and the parameters used for the filter. img_avg is a
        filtered image in the same format as img_in. fltr_params is also
        a list, which contains the final parameters used during the
        interactive session. The first item is the string name of the
        filter that was used, in this case "average". For this function,
        the [fltr_params] list contains:
            
            ["average", (k_size, k_size)]
                k_size: Kernel size for the average filter. If zero, 
                    then no filter was applied.

     ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output. 
    """

    # ---- Start Local Copies ----\
    img = img_in.copy()
    # ---- End Local Copies ----

    # Used to name the new window
    img_title = "Equal-Weighted (Average) Blur Filter" 

    print("\nPress [Enter] while the new window is active to "\
        "continue...")

    cv.namedWindow(img_title, cv.WINDOW_NORMAL) # Create a new window
    cv.moveWindow(img_title, 1, 1) # Move window to top-left of monitor

    # Create interactive trackbars in the new window
    cv.createTrackbar('Kernel Size', img_title, 0, 50, do_nothing)

    # Start interactive session
    while(1):
        # Get current position of the trackbar
        k_size = cv.getTrackbarPos('Kernel Size', img_title)

        # Show original image if kernel size is zero
        if k_size < 1:
            cv.imshow(img_title, img)
            img_avg = img
        else:
            # Perform image processing and show updated image
            img_avg = cv.blur(img, (k_size,k_size))
            cv.imshow(img_title, img_avg)

        # Only way to exit is if a certain key is pressed. Note, 
        # cv.waitKey(50) will force the loop to wait 50 msec to prevent
        # the process thread from running at 100% load. cv.waitKey()
        # returns an integer corresponding to the ASCII code for a key,
        # if it was pressed. Only need the last byte (8 bits) of the 
        # returned 32 bit integer, so a bitwise operation "& 0xFF" is used
        # to get just those necessary 8 bits.
        key_pressed = cv.waitKey(50) & 0xFF

        # Exit if either Enter key is pressed, or the Esc key
        if key_pressed in [27, 10, 13]:
            break

    # Close all opencv figures 
    cv.destroyAllWindows()

    # Apply final set of filter parameters and return the filtered image
    fltr_params = ["average", (k_size, k_size)]

    return [img_avg, fltr_params]


def interact_gaussian_filter(img_in):
    """
    Applies the Gaussian blur filter in an interactive way. The user can
    update the filter's parameters and see the updates in real-time.
    Upon hitting Enter, the interactive window closes, the final image
    is returned, along with the necessary filter parameters to recreate
    the image. These returned parameters work as input to the related
    driver function, apply_driver_blur(...).

     ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

     ---- RETURNED ---- 
    [[img_gauss], [fltr_params]]: Returns a list containing the 
        resultant image and the parameters used for the filter.
        img_gauss is a filtered image in the same format as img_in.
        fltr_params is also a list, which contains the final parameters
        used during the interactive session. The first item is the
        string name of the filter that was used, in this case
        "gaussian". For this  function, the [fltr_params] list
        contains:
            
            ["gaussian", (k_size,k_size), std_dev]
                
                k_size: Kernel size for the Gaussian filter. If zero, 
                    then no filter was applied. Must be odd.
                
                std_dev: Standard deviation for the Gaussian kernel

     ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output. 
    """

    # ---- Start Local Copies ----\
    img = img_in.copy()
    # ---- End Local Copies ----

    # Used to name the new window
    img_title = "Gaussian Blur Filter" 

    print("\nPress [Enter] while the new window is active to "\
        "continue...")

    cv.namedWindow(img_title, cv.WINDOW_NORMAL) # Create a new window
    cv.moveWindow(img_title, 1, 1) # Move window to top-left of monitor

    # Create interactive trackbars in the new window
    cv.createTrackbar('Kernel Size', img_title, 0, 100, do_nothing)
    cv.createTrackbar('Standard Deviation', img_title, 1, 75, do_nothing)
    cv.createTrackbar('If 1: Std Dev is Automatically Calculated', 
        img_title, 0, 1, do_nothing)

    # Change allowable minimum, and initialize the position 
    cv.setTrackbarMin('Standard Deviation', img_title, 1)
    cv.setTrackbarPos('If 1: Std Dev is Automatically Calculated', 
        img_title, 1)

    # Start interactive session
    while(1):
        # Get current position of the trackbars
        k_size = cv.getTrackbarPos('Kernel Size', img_title)
        std_dev = cv.getTrackbarPos('Standard Deviation', img_title)
        auto_std_dev = cv.getTrackbarPos('If 1: Std Dev is Automatically'\
            ' Calculated', img_title)

        # Show original image if kernel size is zero
        if k_size < 1:
            cv.imshow(img_title, img)
            img_gauss = img
        else:
            if k_size % 2 == 0: # Must be odd for this filter
                k_size += -1

            # If true, automatically calculate the standard deviation
            if auto_std_dev:
                std_dev = 0.3*((k_size - 1)*0.5 - 1) + 0.8
                
                # Manually calculating the standard deviation results
                # in a float, so force it back to int for the trackbar
                std_dev_track = int(round(std_dev))

                # Update the trackbar position to inform the user
                cv.setTrackbarPos('Standard Deviation', img_title, 
                    std_dev_track)

            # Perform image processing
            img_gauss = cv.GaussianBlur(img, (k_size,k_size), std_dev)

            # Show updated image
            cv.imshow(img_title, img_gauss)

        # Only way to exit is if a certain key is pressed. Note, 
        # cv.waitKey(50) will force the loop to wait 50 msec to prevent
        # the process thread from running at 100% load. cv.waitKey()
        # returns an integer corresponding to the ASCII code for a key,
        # if it was pressed. Only need the last byte (8 bits) of the 
        # returned 32 bit integer, so a bitwise operation "& 0xFF" is used
        # to get just those necessary 8 bits.
        key_pressed = cv.waitKey(50) & 0xFF

        # Exit if either Enter key is pressed, or the Esc key
        if key_pressed in [27, 10, 13]:
            break

    # Close all opencv figures 
    cv.destroyAllWindows()

    # Apply final set of filter parameters and return the filtered image
    fltr_params = ["gaussian", (k_size,k_size), std_dev]

    return [img_gauss, fltr_params]


def interact_median_filter(img_in):
    """
    Applies the median blur filter in an interactive way. The user can
    update the filter's parameters and see the updates in real-time.
    Upon hitting Enter, the interactive window closes, the final image
    is returned, along with the necessary filter parameters to recreate
    the image. These returned parameters work as input to the related
    driver function, apply_driver_blur(...).

     ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

     ---- RETURNED ---- 
    [[img_med], [fltr_params]]: Returns a list containing the resultant 
        image and the parameters used for the filter. img_med is a
        filtered image in the same format as img_in. fltr_params is also
        a list, which contains the final parameters used during the
        interactive session. The first item is the string name of the
        filter that was used, in this case "median". For this 
        function, the [fltr_params] list contains:
            
            ["median", k_size]
                
                k_size: Kernel size for the median filter. If zero, 
                    then no filter was applied. Must be odd.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output. 
    """

    # ---- Start Local Copies ----\
    img = img_in.copy()
    # ---- End Local Copies ----

    # Used to name the new window
    img_title = "Median Blur Filter" 

    print("\nPress [Enter] while the new window is active to "\
        "continue...")

    cv.namedWindow(img_title, cv.WINDOW_NORMAL) # Create a new window
    cv.moveWindow(img_title, 1, 1) # Move window to top-left of monitor

    # Create interactive trackbars in the new window
    cv.createTrackbar('Kernel Size', img_title, 0, 25, do_nothing)

    # Start interactive session
    while(1):
        # Get current position of the trackbar
        k_size = cv.getTrackbarPos('Kernel Size', img_title)

        if k_size < 1: # Show original image if k_size is zero
            cv.imshow(img_title, img)
            img_med = img
        else:
            if k_size % 2 == 0: # Must be odd for this filter
                k_size += -1

            # Perform image processing and show updated image
            img_med = cv.medianBlur(img, k_size)
            cv.imshow(img_title, img_med)

        # Only way to exit is if a certain key is pressed. Note, 
        # cv.waitKey(50) will force the loop to wait 50 msec to prevent
        # the process thread from running at 100% load. cv.waitKey()
        # returns an integer corresponding to the ASCII code for a key,
        # if it was pressed. Only need the last byte (8 bits) of the 
        # returned 32 bit integer, so a bitwise operation "& 0xFF" is used
        # to get just those necessary 8 bits.
        key_pressed = cv.waitKey(50) & 0xFF

        # Exit if either Enter key is pressed, or the Esc key
        if key_pressed in [27, 10, 13]:
            break

    # Close all opencv figures 
    cv.destroyAllWindows()

    # Apply final set of filter parameters and return the filtered image
    fltr_params = ["median", k_size]

    return [img_med, fltr_params]


def interact_bilateral_filter(img_in):
    """
    Applies the bilateral Gaussian blur filter in an interactive way.
    This filter is an edge-preserving Gaussian blur filter. The user can
    update the filter's parameters and see the updates in real-time.
    Upon hitting Enter, the interactive window closes, the final image
    is returned, along with the necessary filter parameters to recreate
    the image. These returned parameters work as input to the related
    driver function, apply_driver_blur(...).

     ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

     ---- RETURNED ---- 
    [[img_bil], [fltr_params]]: Returns a list containing the resultant 
        image and the parameters used for the filter. img_bil is a
        filtered image in the same format as img_in. fltr_params is also
        a list, which contains the final parameters used during the
        interactive session. The first item is the string name of the
        filter that was used, in this case "bilateral". For this 
        function, the [fltr_params] list contains:
            
            ["bilateral", d_size, sig_intsty]
                
                d_size: Diameter of each pixel neighborhood (must be 
                    even). If zero, then no filter was applied.
                
                sig_intsty: Filter sigma in the color space. A larger
                    value means that farther colors within d_size will 
                    be mixed together, resulting in larger areas of
                    semi-equal color. 
                
                Note: cv.bilateralFilter(...) also takes in a third 
                    parameter that is a filter in coordinate space. 
                    However, since d_size > 0, this third parameter is
                    not used. So, this third parameter is just set equal
                    to sig_intsty in this implementation.

     ---- SIDE EFFECTS ----  
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.  
    """

    # ---- Start Local Copies ----\
    img = img_in.copy()
    # ---- End Local Copies ----

    # Used to name the new window
    img_title = "Bilateral (Edge-Preserving) Blur Filter" 

    print("\nPress [Enter] while the new window is active to "\
        "continue...")

    print("\nWARNING: The bilateral Gaussian filter can be "\
        "computationally costly for large pixel neighborhoods. \n"\
        "For real-time updates, pixel neighborhoods below 7 are "\
        "recommended.")

    cv.namedWindow(img_title, cv.WINDOW_NORMAL) # Create a new window
    cv.moveWindow(img_title, 1, 1) # Move window to top-left of monitor

    # Create interactive trackbars in the new window
    cv.createTrackbar('Pixel Neighborhood', img_title, 0, 15, do_nothing)
    cv.createTrackbar('Intensity Threshold', img_title, 0, 250, 
        do_nothing)

    # Start interactive session
    while(1):
        # Get current position of the trackbar
        d_size = cv.getTrackbarPos('Pixel Neighborhood', img_title)
        sig_intsty = cv.getTrackbarPos('Intensity Threshold', img_title)

        if d_size < 1:
            cv.imshow(img_title, img)
            img_bil = img
        else:
            if d_size % 2 != 0: # If odd
                d_size += 1 # Make it even

            # It seems that this filter only updates if the d_size is
            # even. However, it does not throw an error so long as the
            # d_size is positive and greater than zero.
            img_bil = cv.bilateralFilter(img, d_size, sig_intsty, 
                sig_intsty)

            # Show updated image
            cv.imshow(img_title, img_bil)

        # Only way to exit is if a certain key is pressed. Note, 
        # cv.waitKey(50) will force the loop to wait 50 msec to prevent
        # the process thread from running at 100% load. cv.waitKey()
        # returns an integer corresponding to the ASCII code for a key,
        # if it was pressed. Only need the last byte (8 bits) of the 
        # returned 32 bit integer, so a bitwise operation "& 0xFF" is used
        # to get just those necessary 8 bits.
        key_pressed = cv.waitKey(50) & 0xFF

        # Exit if either Enter key is pressed, or the Esc key
        if key_pressed in [27, 10, 13]:
            break

    # Close all opencv figures 
    cv.destroyAllWindows()

    # Apply final set of filter parameters and return the filtered image
    fltr_params = ["bilateral", d_size, sig_intsty]

    return [img_bil, fltr_params]


def interact_morph(img_in):
    """
    Creates an interactive session that enables the user to apply
    successive morphological operations, like erosion and dilation. The
    user can change the kernel size, ordering of the individual
    erosions/dilations, and the total number of operations. The results
    are updated in real-time. Upon hitting Enter, the interactive window
    closes, and the final image is returned, along with the chosen
    morphological parameters. These parameters can be used as input for
    the related driver function, apply_driver_morph(...).

    ---- INPUT ARGUMENTS ----
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel. 

    ---- RETURNED ---- 
    [[img_morph], [morph_params]]: Returns a list containing the  
        resultant image and the parameters used for the morphological
        operations. img_morph is a filtered image in the same format as
        img_in. morph_params is also a list, which contains the final
        parameters used during the interactive session. In this case,
        the  [morph_params] list contains:
            
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

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output. 
    """
    # ---- Start Local Copies ----
    img = img_in.copy() # Makes a proper (i.e., deep) copy
    # ---- End Start Local Copies ----

    print("\nPress [Enter] while the new window is active to "\
        "continue...")

    img_title = "Open (Denoise): Erosion -> Dilation |"\
        " Close (Holes): Dilation -> Erosion"

    cv.namedWindow(img_title, cv.WINDOW_NORMAL) # Create a new window
    cv.moveWindow(img_title, 1, 1) # Move window to top-left of monitor

    # Create interactive trackbars in the new window
    opn_cls_tbar = "0: Open (Erode -> Dilate)\n1: Close (Dilate -> Erode)\n"
    cv.createTrackbar(opn_cls_tbar, img_title, 0, 1, do_nothing)

    rect_ellps_tbar = "0: Rectangular Kernel\n1: Elliptical Kernel\n"
    cv.createTrackbar(rect_ellps_tbar, img_title, 0, 1, do_nothing)

    cv.createTrackbar("Kernel Width (or Diameter)", img_title, 0, 50, 
        do_nothing)
    cv.setTrackbarPos("Kernel Width (or Diameter)", img_title, 2)

    cv.createTrackbar("Erode Iterations", img_title, 0, 20, do_nothing)
    cv.createTrackbar("Dilate Iterations", img_title, 0, 20, do_nothing)

    # Start interactive session
    while(1):
        # Get current position of the trackbars
        # 0 is open => erosion followed by dilation (removes noise)
        # 1 is close => dilation followed by erosion (closes small holes)
        flag_open_close = cv.getTrackbarPos(opn_cls_tbar, img_title)

        # 0 is rectangular kernel, 1 is elliptical kernel
        flag_rect_ellps = cv.getTrackbarPos(rect_ellps_tbar, img_title)

        # Width (or diameter) of the kernel in pixels
        k_size = cv.getTrackbarPos("Kernel Width (or Diameter)", img_title)

        # Number of iterations of the erode and dilation operations
        num_erode = cv.getTrackbarPos("Erode Iterations", img_title)
        num_dilate = cv.getTrackbarPos("Dilate Iterations", img_title)

        # Example, assume the following parameters:
        #   flag_open_close = 0
        #   num_erode = 3
        #   num_dilate = 2
        #
        # This would lead to the following "open" operation:
        #   erode -> erode -> erode -> dilate -> dilate

        morph_params = [flag_open_close, flag_rect_ellps, k_size, num_erode,
            num_dilate]

        # Perform successive morphological operations in a wrapper function
        img_morph = wrap.multi_morph(img, morph_params)

        # Show the resultant image
        cv.imshow(img_title, img_morph)

        # Only way to exit is if a certain key is pressed. Note, 
        # cv.waitKey(50) will force the loop to wait 50 msec to prevent
        # the process thread from running at 100% load. cv.waitKey()
        # returns an integer corresponding to the ASCII code for a key,
        # if it was pressed. Only need the last byte (8 bits) of the 
        # returned 32 bit integer, so a bitwise operation "& 0xFF" is used
        # to get just those necessary 8 bits.
        key_pressed = cv.waitKey(50) & 0xFF

        # Exit if either Enter key is pressed, or the Esc key
        if key_pressed in [27, 10, 13]:
            break

    # Close all opencv figures 
    cv.destroyAllWindows()

    return [img_morph, morph_params]


def interact_unsharp_mask(img_in):
    """
    Sharpens an image using unsharp masking. This procedure is actually
    implemented manually, since OpenCV does not come with image
    sharpening procedures. This is achieved by subtracting a blurred
    image from the original, and then adding the result to the original
    image by some specified amount (see below), 
        sharpened = original + (original − blurred) X amount 
    The user can update the filter's parameters and see updates in
    real-time. Upon hitting Enter, the interactive window closes, and
    the final image is returned along with the necessary filter
    parameters to recreate the image. These returned parameters work as
    input to the related driver function, apply_driver_sharpen(...).

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

    ---- RETURNED ---- 
    [[img_sharp], [unsharp_params]]: Returns a list containing the 
        resultant image and the parameters used for the unsharp
        procedure.  img_sharp is a sharpened image in the same format as
        img_in. unsharp_params is also a list, which contains the final 
        parameters used during the interactive session. The first entry
        in the parameter list is a string denoting the type of 
        sharpening procedure, which in this case, is "unsharp". The
        remainder of the parameter list is:
            
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

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output. 
    """

    # ---- Start Local Copies ----
    img = img_in.copy()
    # ---- End Local Copies ----

    img_title = "Unsharp Mask for Sharpening Edges"

    print("\nPress [Enter] while the new window is active to "\
        "continue...")

    cv.namedWindow(img_title, cv.WINDOW_NORMAL) # Create a new window
    cv.moveWindow(img_title, 1, 1) # Move window to top-left of monitor

    cv.createTrackbar("Amount [%]", img_title, 0, 500, do_nothing)
    cv.createTrackbar("Pre-Blur Kernel Size", img_title, 0, 100, do_nothing)
    cv.createTrackbar("Pre-Blur Standard Deviation", img_title, 0, 100, 
        do_nothing)
    cv.createTrackbar("If 1: Std Dev is Automatically Calculated", img_title,
        0, 1, do_nothing)
    cv.createTrackbar("0: Pre-Gaussian Filter\n1: Pre-Median Filter\n",  
        img_title, 0, 1, do_nothing)

    cv.setTrackbarPos("Pre-Blur Kernel Size", img_title, 5)
    cv.setTrackbarPos("If 1: Std Dev is Automatically Calculated", 
        img_title, 1)

    # Start interactive session
    while(1):
        cur_amount = cv.getTrackbarPos("Amount [%]", img_title)
        k_size = cv.getTrackbarPos("Pre-Blur Kernel Size", img_title)
        std_dev = cv.getTrackbarPos("Pre-Blur Standard Deviation", img_title)
        auto_std_dev = cv.getTrackbarPos("If 1: Std Dev is Automatically"\
            " Calculated", img_title)
        fltr_type = cv.getTrackbarPos("0: Pre-Gaussian Filter\n"\
            "1: Pre-Median Filter\n", img_title)

        # Must be odd for the Gaussian blur used in the unsharp mask
        if k_size % 2 == 0: 
            k_size += -1

        if auto_std_dev:
            std_dev = 0.3*((k_size - 1)*0.5 - 1) + 0.8
            
            # Manually calculating the standard deviation results
            # in a float, so force it back to int for the trackbar
            std_dev_track = int(round(std_dev))

            # Update the trackbar position to inform the user
            cv.setTrackbarPos("Pre-Blur Standard Deviation", img_title, 
                std_dev_track)

        unsharp_params = [cur_amount, k_size, fltr_type, std_dev]
        img_sharp = wrap.unsharp_mask(img, unsharp_params)

        cv.imshow(img_title, img_sharp)

        # Only way to exit is if a certain key is pressed. Note, 
        # cv.waitKey(50) will force the loop to wait 50 msec to prevent
        # the process thread from running at 100% load. cv.waitKey()
        # returns an integer corresponding to the ASCII code for a key,
        # if it was pressed. Only need the last byte (8 bits) of the 
        # returned 32 bit integer, so a bitwise operation "& 0xFF" is used
        # to get just those necessary 8 bits.
        key_pressed = cv.waitKey(50) & 0xFF

        # Exit if either Enter key is pressed, or the Esc key
        if key_pressed in [27, 10, 13]:
            break

    # Close all opencv figures 
    cv.destroyAllWindows()

    # Add label to the front of the parameter list
    unsharp_params.insert(0, "unsharp")

    return [img_sharp, unsharp_params]


def interact_laplacian_sharp(img_in):
    """
    Sharpens an image similarly to an unsharp mask, but uses the 
    Laplacian to find the edges of the image rather than the difference
    between the original image and the blurred image. This procedure is 
    actually implemented manually, since OpenCV does not come with image
    sharpening procedures. This is achieved as seen below, 
        sharpened = original - (laplacian) X amount 
    Note, however, that the original image is blurred prior to applying
    the Laplacian filter to reduce some of the noise in the image that
    gets accentuated by the Laplacian filter. The user can update the
    filter's parameters and see updates in real-time. Upon hitting
    Enter, the interactive window closes, and the final image is
    returned along with the necessary filter parameters to recreate the
    image. These returned parameters work as input to the related driver
    function, apply_driver_sharpen(...).

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

    ---- RETURNED ---- 
    [[img_sharp], [unsharp_params]]: Returns a list containing the 
        resultant image and the parameters used for the Laplacian
        sharpen procedure. img_sharp is a sharpened image in the same
        format as img_in. unsharp_params is also a list, which contains
        the final  parameters used during the interactive session. The
        first entry in the parameter list is a string denoting the type
        of  sharpening procedure, which in this case, is "laplacian".
        The remainder of the parameter list is:
            
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

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output. 

    """

    # ---- Start Local Copies ----
    img = img_in.copy()
    # ---- End Local Copies ----

    img_title = "Local Laplacian Filter to Sharpen Edges"

    print("\nPress [Enter] while the new window is active to "\
        "continue...")

    cv.namedWindow(img_title, cv.WINDOW_NORMAL) # Create a new window
    cv.moveWindow(img_title, 1, 1) # Move window to top-left of monitor

    cv.createTrackbar("Amount [%]", img_title, 0, 1000, do_nothing)
    cv.createTrackbar("Pre-Blur Kernel Size", img_title, 0, 100, do_nothing)
    cv.createTrackbar("Pre-Blur Standard Deviation", img_title, 0, 100, 
        do_nothing)
    cv.createTrackbar("Laplacian Kernel Size", img_title, 0, 15, do_nothing)
    cv.createTrackbar("If 1: Std Dev is Automatically Calculated", img_title,
        0, 1, do_nothing)
    cv.createTrackbar("0: Pre-Gaussian Filter\n1: Pre-Median Filter\n",  
        img_title, 0, 1, do_nothing)

    cv.setTrackbarPos("Pre-Blur Kernel Size", img_title, 5)
    cv.setTrackbarPos("Laplacian Kernel Size", img_title, 1)
    cv.setTrackbarPos("If 1: Std Dev is Automatically Calculated", 
        img_title, 1)
    # Start interactive session
    while(1):
        cur_amount = cv.getTrackbarPos("Amount [%]", img_title)
        blur_k_size = cv.getTrackbarPos("Pre-Blur Kernel Size", img_title)
        lap_k_size = cv.getTrackbarPos("Laplacian Kernel Size", img_title)
        blur_std_dev = cv.getTrackbarPos("Pre-Blur Standard Deviation", 
            img_title)
        auto_std_dev = cv.getTrackbarPos("If 1: Std Dev is Automatically"\
            " Calculated", img_title)
        fltr_type = cv.getTrackbarPos("0: Pre-Gaussian Filter\n"\
            "1: Pre-Median Filter\n", img_title)

        # Must be odd for the Gaussian blur used in the unsharp mask
        if blur_k_size % 2 == 0: 
            blur_k_size += -1

        if lap_k_size % 2 == 0: 
            lap_k_size += -1

        if auto_std_dev:
            blur_std_dev = 0.3*((blur_k_size - 1)*0.5 - 1) + 0.8
            
            # Manually calculating the standard deviation results
            # in a float, so force it back to int for the trackbar
            std_dev_track = int(round(blur_std_dev))

            # Update the trackbar position to inform the user
            cv.setTrackbarPos("Pre-Blur Standard Deviation", img_title, 
                std_dev_track)

        unsharp_params = [cur_amount, blur_k_size, lap_k_size, fltr_type, 
            blur_std_dev]
        img_sharp = wrap.laplacian_sharp(img, unsharp_params)

        cv.imshow(img_title, img_sharp)

        # Only way to exit is if a certain key is pressed. Note, 
        # cv.waitKey(50) will force the loop to wait 50 msec to prevent
        # the process thread from running at 100% load. cv.waitKey()
        # returns an integer corresponding to the ASCII code for a key,
        # if it was pressed. Only need the last byte (8 bits) of the 
        # returned 32 bit integer, so a bitwise operation "& 0xFF" is used
        # to get just those necessary 8 bits.
        key_pressed = cv.waitKey(50) & 0xFF

        # Exit if either Enter key is pressed, or the Esc key
        if key_pressed in [27, 10, 13]:
            break

    img_sharp = wrap.laplacian_sharp(img, unsharp_params)

    # Close all opencv figures 
    cv.destroyAllWindows()

    # Add label to the front of the list
    unsharp_params.insert(0, "laplacian")

    return [img_sharp, unsharp_params]


def interact_canny_sharp(img_in):
    """
    Sharpens an image similarly to an unsharp mask, but uses the  Canny
    Edge Detection algorithm to find the edges of the image rather than
    the difference between the original image and the blurred image.
    This procedure is  actually implemented manually, since OpenCV does
    not come with image sharpening procedures. This is achieved as seen
    below, 
        sharpened = original - (canny) X amount 
    Note, the Canny Edge Detection algorithm finds edges and marks them
    with discrete lines, so the resultant sharpened image will most
    likely not exhibit smooth gradients in contrast near the edges. The
    user can update the filter's parameters and see updates in
    real-time. Upon hitting Enter, the interactive window closes, and
    the final image is returned along with the necessary filter
    parameters to recreate the image. These returned parameters work as
    input to the related driver function, apply_driver_sharpen(...).

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

    ---- RETURNED ---- 
    [[img_sharp], [unsharp_params]]: Returns a list containing the 
        resultant image and the parameters used for the Canny-edge
        sharpen procedure. img_sharp is a sharpened image in the same
        format as img_in. unsharp_params is also a list, which contains
        the final  parameters used during the interactive session. The
        first entry in the parameter list is a string denoting the type
        of  sharpening procedure, which in this case, is "canny". The
        remainder of the parameter list is:
            
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

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output. 

    """

    # ---- Start Local Copies ----
    img = img_in.copy()
    # ---- End Local Copies ----

    img_title = "Canny Edges Filter to Sharpen Edges"

    print("\nPress [Enter] while the new window is active to "\
        "continue...")

    cv.namedWindow(img_title, cv.WINDOW_NORMAL) # Create a new window
    cv.moveWindow(img_title, 1, 1) # Move window to top-left of monitor

    cv.createTrackbar("Canny Kernel Size", img_title, 0, 7, do_nothing)
    cv.createTrackbar("Amount [%]", img_title, 0, 500, do_nothing)
    cv.createTrackbar("Canny Threshold 1", img_title, 0, 1000, do_nothing)
    cv.createTrackbar("Canny Threshold 2", img_title, 0, 1000, do_nothing)
    #cv.createTrackbar("Threshold", img_title, 0, 255, do_nothing)

    cv.setTrackbarMin("Canny Kernel Size", img_title, 3)
    cv.setTrackbarPos("Amount [%]", img_title, 30)
    cv.setTrackbarPos("Canny Threshold 1", img_title, 200)
    cv.setTrackbarPos("Canny Threshold 2", img_title, 200)

    # Start interactive session
    while(1):
        k_size = cv.getTrackbarPos("Canny Kernel Size", img_title)
        cur_amount = cv.getTrackbarPos("Amount [%]", img_title)
        thresh1 = cv.getTrackbarPos("Canny Threshold 1", img_title)
        thresh2 = cv.getTrackbarPos("Canny Threshold 2", img_title)

        # Must be odd for the Sobel operator
        if k_size % 2 == 0: 
            k_size += -1

        unsharp_params = [k_size, cur_amount, thresh1, thresh2]
        img_sharp = wrap.canny_sharp(img, unsharp_params)

        cv.imshow(img_title, img_sharp)

        # Only way to exit is if a certain key is pressed. Note, 
        # cv.waitKey(50) will force the loop to wait 50 msec to prevent
        # the process thread from running at 100% load. cv.waitKey()
        # returns an integer corresponding to the ASCII code for a key,
        # if it was pressed. Only need the last byte (8 bits) of the 
        # returned 32 bit integer, so a bitwise operation "& 0xFF" is used
        # to get just those necessary 8 bits.
        key_pressed = cv.waitKey(50) & 0xFF

        # Exit if either Enter key is pressed, or the Esc key
        if key_pressed in [27, 10, 13]:
            break

    # Close all opencv figures 
    cv.destroyAllWindows()

    # Add label to the front of the list
    unsharp_params.insert(0, "canny")

    return [img_sharp, unsharp_params]


def interact_adaptive_equalize(img_in):
    """
    Creates an interactive session to equalize the histogram of an 
    image using Contrast Limited Adaptive Histogram Equalization 
    (CLAHE). This procedure is an locally adaptive and also applies
    contrast limiting to help prevent the amplification of noise. The
    resultant image will have locally enhanced contrast, which can
    amplify noise despite the contrast limiting feature. The user can
    update the equalization parameters and see updates in real-time.
    Upon hitting Enter, the interactive window closes, and the final
    image is returned along with the necessary filter parameters to
    recreate the image. These returned parameters work as input to the
    related driver function, apply_driver_equalize(...).

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

    ---- RETURNED ----
    [[img_eq], [eq_params]]: Returns a list containing the resultant 
        image and the parameters used for the equalization procedure.
        img_eq  is an image with locally enhanced contrast that is in
        the same format as img_in. eq_params is also a list, which
        contains the final parameters used during the interactive
        session. The first entry in the parameter list is a string
        denoting the type of equalization procedure used, which in this
        case, is "adaptive". The reaminder of the parameter list is:
            
            ["adaptive", clip_limit, grid_size]
                
                clip_limit: Threshold for contrast limiting
                
                grid_size: Size of grid for histogram equalization. 
                    Input image will be divided into equally sized 
                    rectangular tiles. tileGridSize defines the number 
                    of tiles in row and column. If zero, the original
                    image is returned.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """

    # ---- Start Local Copies ----
    img = img_in.copy()
    # ---- End Local Copies ----

    img_title = "Contrast Limited Adaptive Histogram Equalization"

    print("\nPress [Enter] while the new window is active to "\
        "continue...")
    
    cv.namedWindow(img_title, cv.WINDOW_NORMAL) # Create a new window
    cv.moveWindow(img_title, 1, 1) # Move window to top-left of monitor

    cv.createTrackbar("Contrast Clip Limit", img_title, 0, 50, do_nothing)
    cv.createTrackbar("Tile Grid Size", img_title, 0, 50, do_nothing)

    cv.setTrackbarPos("Contrast Clip Limit", img_title, 2)
    cv.setTrackbarMin("Contrast Clip Limit", img_title, 1)
    cv.setTrackbarPos("Tile Grid Size", img_title, 8)

    # Start interactive session
    while(1):
        clip_limit = cv.getTrackbarPos("Contrast Clip Limit", img_title)
        grid_size = cv.getTrackbarPos("Tile Grid Size", img_title)

        eq_params = [clip_limit, grid_size]
        img_eq = wrap.adaptive_equalize(img, eq_params)

        cv.imshow(img_title, img_eq)

        # Only way to exit is if a certain key is pressed. Note, 
        # cv.waitKey(50) will force the loop to wait 50 msec to prevent
        # the process thread from running at 100% load. cv.waitKey()
        # returns an integer corresponding to the ASCII code for a key,
        # if it was pressed. Only need the last byte (8 bits) of the 
        # returned 32 bit integer, so a bitwise operation "& 0xFF" is used
        # to get just those necessary 8 bits.
        key_pressed = cv.waitKey(50) & 0xFF

        # Exit if either Enter key is pressed, or the Esc key
        if key_pressed in [27, 10, 13]:
            break

    # Close all opencv figures 
    cv.destroyAllWindows()

    # Add a label to the front of the parameter list
    eq_params.insert(0, "adaptive")

    return [img_eq, eq_params]


def interact_global_threshold(img_in):
    """
    Creates an interactive session to threshold a grayscale image into a
    black and white image (also known as binarizing an image). This 
    procedure allows the user to pick the intensity (from 0 to 255) that
    gets used to threshold the entire image. Hence, this is a global
    thresholding procedure. Alternatively, the thresholding value can be
    automatically chosen by the program based on Otsu's method. The
    resultant image will contain just black and white values, which
    corresponds to 0 and 255, respectively. Upon hitting Enter, the
    interactive window closes, and the final image is returned along
    with the necessary threshold parameters to recreate the image. These
    returned parameters work as input to the related driver function,
    apply_driver_thresh(...).

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

    ---- RETURNED ----
    [[img_thsh], [thsh_params]]: Returns a list containing the resultant
        image and the parameters used for the thresholding procedure.
        img_thsh is a binarized image that is in the same format as
        img_in. thsh_params is also a list, which contains the final
        parameters used during the interactive session. The first entry
        in the parameter list is a string denoting the type of
        thresholding procedure used, which in this case, is "global".
        The remainder of the parameter list is:
            
            ["global", cur_thsh]
                
                cur_thsh: The specified intensity to use as the 
                    threshold limit between black and white. It should
                    be an integer between 0 and 255. However, if
                    cur_thsh < 0, then cur_thsh will be calculated
                    automatically using Otsu's method.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output. 
    """

    img = img_in.copy()

    # Used to name the new window
    img_title = "Global Threshold"

    print("\nPress [Enter] while the new window is active to "\
        "continue...")

    cv.namedWindow(img_title, cv.WINDOW_NORMAL) # Create a new window
    cv.moveWindow(img_title, 1, 1) # Move window to top-left of monitor

    # Create interactive trackbars in the new window
    cv.createTrackbar("Threshold Intensity", img_title, 0, 255, do_nothing)
    cv.createTrackbar("If 0: Manual Control\nIf 1: Automatic (Otsu)\n", 
        img_title, 0, 1, do_nothing)

    # Initialize threshold value using the Otsu method
    [ret_thsh, img_thsh] = cv.threshold(img, 0, 255, 
                cv.THRESH_BINARY+cv.THRESH_OTSU)
    thsh_track = int(round(ret_thsh))
    cv.setTrackbarPos("Threshold Intensity", img_title, thsh_track)

    # Start interactive session
    while(1):
        cur_thsh = cv.getTrackbarPos("Threshold Intensity", img_title)
        auto_otsu = cv.getTrackbarPos("If 0: Manual Control\n"\
            "If 1: Automatic (Otsu)\n", img_title)

        if auto_otsu:
            [ret_thsh, img_thsh] = cv.threshold(img, 0, 255, 
                cv.THRESH_BINARY+cv.THRESH_OTSU)

            cur_thsh = ret_thsh
            thsh_track = int(round(ret_thsh))
            cv.setTrackbarPos("Threshold Intensity", img_title, thsh_track)
            
        else:
            [ret_thsh, img_thsh] = cv.threshold(img, cur_thsh, 255, 
                cv.THRESH_BINARY)

        cv.imshow(img_title, img_thsh)

        # Only way to exit is if a certain key is pressed. Note, 
        # cv.waitKey(50) will force the loop to wait 50 msec to prevent
        # the process thread from running at 100% load. cv.waitKey()
        # returns an integer corresponding to the ASCII code for a key,
        # if it was pressed. Only need the last byte (8 bits) of the 
        # returned 32 bit integer, so a bitwise operation "& 0xFF" is used
        # to get just those necessary 8 bits.
        key_pressed = cv.waitKey(50) & 0xFF

        # Exit if either Enter key is pressed, or the Esc key
        if key_pressed in [27, 10, 13]:
            break

    # Close opencv windows
    cv.destroyAllWindows()

    thsh_params = ["global", cur_thsh]
    return [img_thsh, thsh_params]


def interact_mean_threshold(img_in):
    """
    Creates an interactive session to threshold a grayscale image into a
    black and white image (also known as binarizing an image). This
    thresholding procedure is a locally adaptive method based on the
    mean kernel. The resultant image will contain just black and white
    values, which corresponds to 0 and 255, respectively. Upon hitting
    Enter, the interactive window closes, and the final image is
    returned along with the necessary threshold parameters to recreate
    the image. These returned parameters work as input to the related
    driver function, apply_driver_thresh(...).

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

    ---- RETURNED ----
    [[img_thsh], [thsh_params]]: Returns a list containing the resultant 
        image and the parameters used for the thresholding procedure.
        img_thsh is a binarized image that is in the same format as
        img_in. thsh_params is also a list, which contains the final
        parameters used during the interactive session. The first entry
        in the parameter list is a string denoting the type of
        thresholding procedure used, which in this case, is 
        "adaptive_mean". The remainder of the parameter list is:
            
            ["adaptive_mean", blk_size, c_offset]
                
                blk_size: Size of a pixel neighborhood that is used to
                    calculate a threshold value for the current pixel.
                    Must be odd, and if zero, no thresholding is done.
                
                c_offset: A constant offset value applied to the mean
                    or weighted mean of the intensities

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """

    img = img_in.copy()

    # Used to name the new window
    img_title = "Adaptive Mean Threshold"

    print("\nPress [Enter] while the new window is active to "\
        "continue...")

    cv.namedWindow(img_title, cv.WINDOW_NORMAL) # Create a new window
    cv.moveWindow(img_title, 1, 1) # Move window to top-left of monitor

    # Create interactive trackbars in the new window
    cv.createTrackbar("Block Size", img_title, 0, 200, do_nothing)
    cv.createTrackbar("Constant Offset", img_title, 0, 200, do_nothing)

    cv.setTrackbarPos("Block Size", img_title, 11)
    cv.setTrackbarMin("Constant Offset", img_title, -200)

    # Create interactive trackbars in the new window
    while(1):
        blk_size = cv.getTrackbarPos("Block Size", img_title)
        c_offset = cv.getTrackbarPos("Constant Offset", img_title)

        if blk_size <= 0:
            cv.imshow(img_title, img)
            img_thsh = img

        else:
            if blk_size % 2 == 0: # Must be odd pixel neighborhood size
                blk_size += -1

            if blk_size < 3: # Only odd numbers, but also does not like 1
                blk_size = 3

            img_thsh = cv.adaptiveThreshold(img, 255, 
                cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 
                blk_size, c_offset)

            cv.imshow(img_title, img_thsh)

        # Only way to exit is if a certain key is pressed. Note, 
        # cv.waitKey(50) will force the loop to wait 50 msec to prevent
        # the process thread from running at 100% load. cv.waitKey()
        # returns an integer corresponding to the ASCII code for a key,
        # if it was pressed. Only need the last byte (8 bits) of the 
        # returned 32 bit integer, so a bitwise operation "& 0xFF" is used
        # to get just those necessary 8 bits.
        key_pressed = cv.waitKey(50) & 0xFF

        # Exit if either Enter key is pressed, or the Esc key
        if key_pressed in [27, 10, 13]:
            break

    # Close opencv windows
    cv.destroyAllWindows()

    thsh_params = ["adaptive_mean", blk_size, c_offset]
    return [img_thsh, thsh_params]


def interact_gaussian_threshold(img_in):
    """
    Creates an interactive session to threshold a grayscale image into a
    black and white image (also known as binarizing an image). This
    thresholding procedure is a locally adaptive method based on the
    Gaussian kernel. The resultant image will contain just black and
    white values, which corresponds to 0 and 255, respectively. Upon
    hitting Enter, the interactive window closes, and the final image is
    returned along with the necessary threshold parameters to recreate
    the image. These returned parameters work as input to the related
    driver function, apply_driver_thresh(...).

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

    ---- RETURNED ----
    [[img_thsh], [thsh_params]]: Returns a list containing the resultant 
        image and the parameters used for the thresholding procedure.
        img_thsh is a binarized image that is in the same format as
        img_in. thsh_params is also a list, which contains the final
        parameters used during the interactive session. The first entry
        in the parameter list is a string denoting the type of
        thresholding procedure used, which in this case, is 
        "adaptive_ gaussian". The remainder of the parameter list is:
            
            ["adaptive_gaussian", blk_size, c_offset]
                
                blk_size: Size of a pixel neighborhood that is used to
                    calculate a threshold value for the current pixel.
                    Must be odd, and if zero, no thresholding is done.
                
                c_offset: A constant offset value applied to the mean
                    or weighted mean of the intensities

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.

    """

    img = img_in.copy()

    # Used to name the new window
    img_title = "Adaptive Gaussian Threshold"

    print("\nPress [Enter] while the new window is active to "\
        "continue...")

    cv.namedWindow(img_title, cv.WINDOW_NORMAL) # Create a new window
    cv.moveWindow(img_title, 1, 1) # Move window to top-left of monitor

    # Create interactive trackbars in the new window
    cv.createTrackbar("Block Size", img_title, 0, 200, do_nothing)
    cv.createTrackbar("Constant Offset", img_title, 0, 200, do_nothing)

    cv.setTrackbarPos("Block Size", img_title, 11)
    cv.setTrackbarMin("Constant Offset", img_title, -200)

    while(1):
        blk_size = cv.getTrackbarPos("Block Size", img_title)
        c_offset = cv.getTrackbarPos("Constant Offset", img_title)

        if blk_size <= 0:
            cv.imshow(img_title, img)
            img_thsh = img

        else:
            if blk_size % 2 == 0: # Must be odd pixel neighborhood size
                blk_size += -1

            if blk_size < 3: # Only odd numbers, but also does not like 1
                blk_size = 3

            img_thsh = cv.adaptiveThreshold(img, 255, 
                cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 
                blk_size, c_offset)

            cv.imshow(img_title, img_thsh)

        # Only way to exit is if a certain key is pressed. Note, 
        # cv.waitKey(50) will force the loop to wait 50 msec to prevent
        # the process thread from running at 100% load. cv.waitKey()
        # returns an integer corresponding to the ASCII code for a key,
        # if it was pressed. Only need the last byte (8 bits) of the 
        # returned 32 bit integer, so a bitwise operation "& 0xFF" is used
        # to get just those necessary 8 bits.
        key_pressed = cv.waitKey(50) & 0xFF

        # Exit if either Enter key is pressed, or the Esc key
        if key_pressed in [27, 10, 13]:
            break

    # Close opencv windows
    cv.destroyAllWindows()

    thsh_params = ["adaptive_gaussian", blk_size, c_offset]
    return [img_thsh, thsh_params]


def interact_blob_fill(img_in):
    """
    Removes blobs (i.e., islands) of pixels from a binarized image in an
    interactive sesssion. Blobs are identified based on user-selected
    thresholding parameters related to the blob size (in pixels),
    circularity, and aspect ratio. It is assumed that the blobs to be
    removed are white, and thus, will be filled in with black pixels to
    effectively delete them. Upon hitting Enter during the interactive
    session, the interactive window closes and the final image is
    returned along with the necessary parameters to recreate the image.
    These returned parameters work as input to the related driver
    function, apply_driver_blob_fill(...).

    ---- INPUT ARGUMENTS ----
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

    ---- RETURNED ---- 
    [[img], [blob_params]]: Returns a list containing the resultant 
        image and the parameters used for the blob-deletion procedure.
        img is the resultant image after certain blobs have been deleted
        (by filling them in with black pixels). blob_params is also a 
        list, which contains the final parameters used during the 
        interactive session. More details about the parameters in
        blob_params is given below:
            
            [area_thresh_min, area_thresh_max, circty_thresh_min,
            circty_thresh_max, ar_min, ar_max, BLACK]
                
                area_thresh_min: Lower threshold bound for identifying
                    blobs to be deleted. Related to the blob size based
                    on the total number of pixels.
                
                area_thresh_max: Upper threshold bound for identifying
                    blobs to be deleted. Related to the blob size based
                    on the total number of pixels.
                
                circty_thresh_min: Lower threshold bound for identifying
                    blobs to be deleted. Related to the circularity of
                    the blob, based on, 4*pi*area/(perimeter)^2
                
                circty_thresh_max: Upper threshold bound for identifying
                    blobs to be deleted. Related to the circularity of
                    the blob, based on, 4*pi*area/(perimeter)^2
                
                ar_min: Lower threshold bound for identifying blobs to 
                    be deleted. Related to the aspect ratio, which is
                    the width over length of the bounding rectangle.
                
                ar_max: Upper threshold bound for identifying blobs to 
                    be deleted. Related to the aspect ratio, which is
                    the width over length of the bounding rectangle.
                
                BLACK: The color used to fill in the pixels of each
                    blob. Should be a tuple containing three integers,
                    each corresponding to the intensity of the color
                    channels (in BGR format). This function always fills
                    pixels in with black, so this parameter is always
                    (0, 0, 0).

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """

    img = img_in.copy() # Copy of input binarized image
    PI = float(3.14159265358979323846264338327950288419716939937510)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    img_title = "Blob Fill"

    print("\nPress [Enter] while the new window is active to "\
        "continue...")

    cv.namedWindow(img_title, cv.WINDOW_NORMAL) # Create a new window
    cv.moveWindow(img_title, 1, 1) # Move window to top-left of monitor

    # Create interactive trackbars in the new window
    cv.createTrackbar("Min. Area Threshold", img_title, 0, 100, do_nothing)
    cv.createTrackbar("Max. Area Threshold", img_title, 0, 300, do_nothing)
    cv.createTrackbar("Min. Circularity [%]", img_title, 0, 100, do_nothing)
    cv.createTrackbar("Max. Circularity [%]", img_title, 0, 110, do_nothing)
    cv.createTrackbar("Min. Aspect Ratio [%]", img_title, 0, 500, do_nothing)
    cv.createTrackbar("Max. Aspect Ratio [%]", img_title, 0, 1000, do_nothing)

    cv.setTrackbarPos("Max. Area Threshold", img_title, 10)
    cv.setTrackbarPos("Max. Circularity [%]", img_title, 110)
    cv.setTrackbarPos("Max. Aspect Ratio [%]", img_title, 1000)

    [img_contrs, contr_hierarchy] = cv.findContours(img, cv.RETR_LIST, 
        cv.CHAIN_APPROX_SIMPLE)

    # Create interactive trackbars in the new window
    while(1):

        contrs_area = []
        contrs_perim = []
        contrs_circty = []
        contrs_del = []

        area_thresh_min = cv.getTrackbarPos("Min. Area Threshold", img_title) 
        area_thresh_max = cv.getTrackbarPos("Max. Area Threshold", img_title) 
        circty_thresh_min = cv.getTrackbarPos("Min. Circularity [%]", 
            img_title)
        circty_thresh_max = cv.getTrackbarPos("Max. Circularity [%]", 
            img_title)
        ar_min = cv.getTrackbarPos("Min. Aspect Ratio [%]", img_title)
        ar_max = cv.getTrackbarPos("Max. Aspect Ratio [%]", img_title)

        area_thresh_min = int(round(area_thresh_min)) # In pixels
        area_thresh_max = int(round(area_thresh_max))
        circty_thresh_min = circty_thresh_min/100.0
        circty_thresh_max = circty_thresh_max/100.0
        ar_min = ar_min/100.0
        ar_max = ar_max/100.0

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

        if img_contrs:
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
                cv.drawContours(img, [contr], 0, BLACK, thickness=cv.FILLED, 
                    lineType=cv.LINE_8)

        cv.imshow(img_title, img)        

        # Only way to exit is if a certain key is pressed. Note, 
        # cv.waitKey(50) will force the loop to wait 50 msec to prevent
        # the process thread from running at 100% load. cv.waitKey()
        # returns an integer corresponding to the ASCII code for a key,
        # if it was pressed. Only need the last byte (8 bits) of the 
        # returned 32 bit integer, so a bitwise operation "& 0xFF" is used
        # to get just those necessary 8 bits.
        key_pressed = cv.waitKey(50) & 0xFF

        # Exit if either Enter key is pressed, or the Esc key
        if key_pressed in [27, 10, 13]:
            break

        # Reset the image, otherwise, the blobs never get un-deleted
        # when the user reduces the thresholds
        img = img_in.copy()

    # Close opencv windows
    cv.destroyAllWindows()

    blob_params = [area_thresh_min, area_thresh_max, circty_thresh_min,
        circty_thresh_max, ar_min, ar_max, BLACK]

    return [img, blob_params]


def interact_denoise(img_in):
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
    returned along with the necessary threshold parameters to recreate
    the image. These returned parameters work as input to the related
    driver function, apply_driver_denoise(...).

    ---- INPUT ARGUMENTS ----
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel. 

    ---- RETURNED ---- 
    [[img_denoise], [denoise_params]]: Returns a list containing the 
        resultant image and the parameters used for the denoising 
        procedure. img_denoise is the resultant image after denoising.
        denoise_params is also a list, which contains the final 
        parameters used during the interactive session. More 
        specifically, the parameters in denoise_params are,
            
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

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output.
    """
    

    img = img_in.copy()

    img_title = "Non-Local Means Denoising"

    print("\nPress [Enter] while the new window is active to "\
        "continue...")

    print("\nWARNING: This non-local means denoising filter is very "\
        "computationally costly!\n    The cost scales approximately linearly"\
        " for increasing values of\n    Window Size and Template Patch Size.")

    cv.namedWindow(img_title, cv.WINDOW_NORMAL) # Create a new window
    cv.moveWindow(img_title, 1, 1) # Move window to top-left of monitor

    # Create interactive trackbars in the new window
    cv.createTrackbar("Filter Strength", img_title, 0, 50, do_nothing)
    cv.createTrackbar("Template Patch Size", img_title, 0, 25, do_nothing)
    cv.createTrackbar("Window Size", img_title, 0, 51, do_nothing)

    cv.setTrackbarPos("Filter Strength", img_title, 3)
    cv.setTrackbarPos("Template Patch Size", img_title, 7)
    cv.setTrackbarPos("Window Size", img_title, 21)

    # Create interactive trackbars in the new window
    while(1):

        cur_h = cv.getTrackbarPos("Filter Strength", img_title)
        cur_tsize = cv.getTrackbarPos("Template Patch Size", img_title)
        cur_wsize = cv.getTrackbarPos("Window Size", img_title)

        if cur_h <= 0:
            cv.imshow(img_title, img)
            img_denoise = img

        else:
            # Must be odd for
            if cur_tsize % 2 == 0: 
                cur_tsize += -1

            if cur_wsize % 2 == 0: 
                cur_wsize += -1

            img_denoise = cv.fastNlMeansDenoising(img, h=cur_h, 
                templateWindowSize=cur_tsize, searchWindowSize=cur_wsize)

            cv.imshow(img_title, img_denoise)

        # Only way to exit is if a certain key is pressed. Note, 
        # cv.waitKey(200) will force the loop to wait 200 msec to prevent
        # the process thread from running at 100% load. cv.waitKey()
        # returns an integer corresponding to the ASCII code for a key,
        # if it was pressed. Only need the last byte (8 bits) of the 
        # returned 32 bit integer, so a bitwise operation "& 0xFF" is used
        # to get just those necessary 8 bits.
        key_pressed = cv.waitKey(200) & 0xFF

        # Exit if either Enter key is pressed, or the Esc key
        if key_pressed in [27, 10, 13]:
            break

    # Close opencv windows
    cv.destroyAllWindows()

    denoise_params = [cur_h, cur_tsize, cur_wsize]

    return [img_denoise, denoise_params]