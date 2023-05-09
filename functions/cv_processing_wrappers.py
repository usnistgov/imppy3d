import numpy as np
import cv2 as cv


def crop_img(img_in, roi_in, quiet_in=False):
    """
    Returns a cropped portion of the original image, which is positioned
    about the center of the original image. In other words, the anchor
    for the cropping operation is always located on the center pixel
    of the original image.

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

    (roi_in): A tuple of two or four integers representing:
        Option a (centered crop, length 2): the number of rows and columns
          of pixels to be retained in the final image. The first
          integer corresponds to the number of rows, and the second to
          columns. Use negative numbers to prevent cropping in that 
          dimension. For example, (-1, 100) would keep all of the rows of
          the original image and crop the number of columns to a 100 
          pixels. Integers can be even or odd, and the cropping operation
          is always anchored about the center pixel of the original image.
        Option b (arbitrary crop, length 4): works just like array slicing
          where image is cropped rectangularly following bounds given as
          (row_min, row_max, column_min, column_max).

    quiet_in: A boolean that determiens if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing

    ---- RETURNED ---- 
    [img_crop]: The resultant image after cropping. img_crop is in the
        same format as img_in.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. Strings are 
    printed to standard output.
    """

    # ---- Start Local Copies ----
    img = img_in.copy() # Makes a proper (i.e., deep) copy
    roi = list(roi_in) # Region-of-interest (ROI)
    quiet = quiet_in
    # ---- End Start Local Copies ----

    # If not already, force the data to be of type integer
    for count, element in enumerate(roi):
        roi[count] = int(element)

    img_shape = img.shape # Tuple containing the number of rows and columns
    
    if len(roi) == 2: # Use a center anchor and number of rows and columns in roi
        
        if (roi[0] > img_shape[0]) or (roi[1] > img_shape[1]):
            if not quiet:
                print(f"\nWARNING: Cropped region is larger than the original "\
                    f"\nimage. Original Image Shape: {img_shape}"\
                    f"\nInput Crop Parameters:{roi}"\
                    f"\nUsing the original image size.")
            return img

        elif (roi[0] <= 0) and (roi[1] <= 0): # No cropping desired
            return img 

        # Keep the entire row or column if this is less than or equal to zero
        if roi[0] <= 0:
            roi[0] = img_shape[0]
        if roi[1] <= 0:
            roi[1] = img_shape[1]

        # Even integers in the ROI require special treatment
        r_even, c_even = False, False
        if roi[0] % 2 == 0:
            r_even = True
            roi[0] -= 1 # Make it odd for now, and correct for it later

        if roi[1] % 2 == 0:
            c_even = True
            roi[1] -= 1 # Make it odd for now, and correct for it later

        # Get the middle (row, col) indices of the original image. 
        if img_shape[0] % 2 == 0: # If even
            mid_r_index = int(img_shape[0]/2) - 1 # -1 b/c indices start at 0
        else: # Else odd
            mid_r_index = int(img_shape[0]/2) # int() rounds down, so OK

        if img_shape[1] % 2 == 0: # If even
            mid_c_index = int(img_shape[1]/2) - 1 # -1 b/c indices start at 0
        else: # Else odd
            mid_c_index = int(img_shape[1]/2) # int() rounds down, so OK

        mid_indices = (mid_r_index, mid_c_index) 

        # Number of pixels to keep above and below (or left and right of) the  
        # center. Should not need int(), but just in case.
        r_radius = int((roi[0] - 1)/2) # (Odd_Number - 1)/2 
        c_radius = int((roi[1] - 1)/2) # (Odd_Number - 1)/2 

        # List of lower/upper bounds on the indices to be kept
        r_bounds = [mid_indices[0] - r_radius, mid_indices[0] + r_radius + 1]
        c_bounds = [mid_indices[1] - c_radius, mid_indices[1] + c_radius + 1]

        # Middle coordinate was subtracted by one if even, so add extra index on
        # the upper end to ensure the correct length in the ROI
        if r_even: # If originally had even number of rows in ROI
            r_bounds[1] = r_bounds[1] + 1
        if c_even: # If originally had even number of columns in ROI
            c_bounds[1] = c_bounds[1] + 1
        crop_type = 'Centered'
        
    elif len(roi) == 4: # roi defines rmin,rmax,cmin,cmax
        # Check that the difference between min and max is less than or 
        # equal to the img size, and also that the max is smaller than or
        # equal to the image size.
        if ( (roi[1] - roi[0]) > img_shape[0]) or ( (roi[3] - roi[2]) > img_shape[1])\
            or (roi[1] > img_shape[0]) or (roi[3] > img_shape[1]):
            if not quiet:
                print(f"\nWARNING: Cropped region is larger than the original "\
                    f"\nimage. Original Image Shape: {img_shape}"\
                    f"\nInput Crop Parameters:{roi}"\
                    f"\nUsing the original image size.")
            return img

        elif (roi[0] <= 0) and (roi[1] <= 0) and (roi[2] <= 0) and (roi[3] <= 0): # No cropping desired
            return img 
    
        r_bounds = [roi[0], roi[1]]
        c_bounds = [roi[2], roi[3]]
        crop_type = 'Rectangle'
        
    else:
        print(f"\nWARNING: ROI parameters malformed (too few or too many)."
                    f"\nInput Crop Parameters:{roi}"\
                    f"\nUsing the original image size.")
        return img
        
    # Perform the actual slicing operation to crop the original image
    img_crop = img[r_bounds[0]:r_bounds[1], c_bounds[0]:c_bounds[1]]

    if not quiet:
        print("\nSuccessfully cropped the image:\n"\
            f"    Crop type: {crop_type}\n"\
            f"    Number of Rows: {img_crop.shape[0]} pixels\n"\
            f"    Number of Columns: {img_crop.shape[1]} pixels")

    # Return the cropped image
    return img_crop


def normalize_histogram(img_in, bounds_in=(0,255), quiet_in=False):
    """
    Perform a linear normalization of a grayscale digital image, also
    known as histogram stretching. The minimum pixel intensity will be 
    mapped to 0, and maximum pixel intensity to 255.

    ---- INPUT ARGUMENTS ----
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

    (bounds_in): A tuple containing two integers that must range between
        0 and 255. The first integer is the intensity lower-bound, and
        the second integer is the intensity upper-bound. These intensity
        bounds will be used to truncate the original image's histogram.
        After normalization, any pixel that was originally below the 
        lower-bound will be black (i.e., 0). Conversely, any pixel that
        was originally above the upper-bound will be white (i.e., 255).

    quiet_in: A boolean that determiens if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing

    ---- RETURNED ----
    [img_new]: Same data structure as the input, img_in. However, the 
        returned image numpy array is stretched so that the intensity
        values span from 0 to 255.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. Strings are 
    printed to standard output.
    """

    # ---- Start Local Copies ----
    img = img_in.copy() # Makes a proper (i.e., deep) copy
    low_bound = bounds_in[0]
    up_bound = bounds_in[1]
    quiet = quiet_in
    # ---- End Start Local Copies ----

    if low_bound == up_bound:
        low_bound = 0
        up_bound = 255

    elif low_bound > up_bound:
        temp_bound = low_bound
        low_bound = up_bound
        up_bound = temp_bound

    if up_bound > 255:
        up_bound = 255

    if low_bound < 0:
        low_bound = 0

    # Effectively threshold the upper and lower bounds of the image's
    # histogram based on the provided inputs
    img[img < low_bound] = low_bound
    img[img > up_bound] = up_bound

    # This is just a new view into img, does not make an actual memory copy
    img_new = img.copy()

    # OpenCV's normalize function will do this automatically
    cv.normalize(img, img_new, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    mean1_intensity = cv.mean(img_in)
    mean2_intensity = cv.mean(img_new)

    if not quiet:
        print("\nSuccessfully normalized the intensity histogram from"\
            " 0 to 255:\n"\
            f"    Mean Intensity of Original Image: {mean1_intensity[0]}\n"\
            f"    Mean Intensity of Normalized Image: {mean2_intensity[0]}")

    return img_new


def multi_morph(img_in, morph_params_in):
    """
    Applies a morphological operation (i.e., dilation and erosion
    transformations). Depending on the input parameters, this function
    can either perform just dilation transformation, or just erosion
    transformations. Alternatively, the user can perform either an open
    transformation or a close transformation. Note, an open
    transformation is some number of erosion-steps followed by some
    number of dilation-steps, and it can be helpful in removing noise.
    Conversely, a close transformation is  some number of dilation-steps
    followed by some number of erosion-steps, and it can be useful in
    closing small holes.

    ---- INPUT ARGUMENTS ----
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

    [morph_params_in]: A list of parameters necessary to define the
        morphological operation to be applied. A more detailed
        explanation of each parameter in morph_params_in is given
        below,
            
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

    ---- RETURNED ----
    [img_morph]: The resultant image after various erosion and dilation
        transformations is returned. The returned image has the same 
        data structure and data type as img_in.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function.
    """

    # ---- Start Local Copies ----
    img = img_in.copy()
    morph_params = morph_params_in
    # ---- End Local Copies ----

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

    # Show original image if kernel size is less than one, or the number
    # of iterations for both erosion and dilation are zero
    if k_size < 1:
        return img
    elif (num_dilate < 1) and (num_erode < 1):
        return img

    # Create the kernel
    if flag_rect_ellps: # Elliptical
        cur_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, 
            (k_size,k_size))
    else: # Rectangular
        cur_kernel = cv.getStructuringElement(cv.MORPH_RECT, 
            (k_size,k_size))

    # Do the successive morphological operations
    if flag_open_close: # Close
        if (num_dilate >= 1) and (num_erode >= 1):
            img_morph = cv.dilate(img, cur_kernel, 
                iterations=num_dilate)
            img_morph = cv.erode(img_morph, cur_kernel, 
                iterations=num_erode)

        elif num_dilate < 1: # Only do a erosion
            img_morph = cv.erode(img, cur_kernel, 
                iterations=num_erode)

        else: # Only do an dilation
            img_morph = cv.dilate(img, cur_kernel, 
                iterations=num_dilate)

    else: # Open
        if (num_dilate >= 1) and (num_erode >= 1):
            img_morph = cv.erode(img, cur_kernel, 
                iterations=num_erode)
            img_morph = cv.dilate(img_morph, cur_kernel, 
                iterations=num_dilate)

        elif num_dilate < 1: # Only do a erosion
            img_morph = cv.erode(img, cur_kernel, 
                iterations=num_erode)

        else: # Only do an dilation
            img_morph = cv.dilate(img, cur_kernel, 
                iterations=num_dilate)

    return img_morph


def unsharp_mask(img_in, unsharp_params_in):
    """
    Sharpens an image using unsharp masking. This is achieved by
    subtracting a blurred image from the original, and then adding the
    result to the original image by some specified amount (see below), 
        sharpened = original + (original − blurred) X amount

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

    [unsharp_params_in]: A list of parameters needed to perform the
        edge enhancement effect. Details of each parameter in 
        unsharp_params_in is given below,

            [cur_amount, k_size, fltr_type, std_dev]

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

    ---- RETURNED ---- 
    [img_sharp]: Returns the resultant image after performing the
        image processing procedures. img_sharp is in the same format as
        img_in.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function.
    """

    # ---- Start Local Copies ----
    img = img_in.copy()
    unsharp_params = unsharp_params_in
    # ---- End Local Copies ----

    # As a percentage, this is a weight for adding images together. A 
    # typical value would be between 50% and 150%. Larger amount increases
    # the sharpening effect
    amount = unsharp_params[0]/100.0 # Convert out of a percentage

    # Related to kernel size of the blur filter. This affects the edge
    # thickness of the sharpened image
    radius = unsharp_params[1] # In pixels
    if radius % 2 == 0: # Must be odd for the blur filters
        radius += -1

    # Affects which blur filter is used. 0 does a Gaussian blur, while 1
    # does a median blur, which is more edge-preserving.
    fltr_type = unsharp_params[2]

    # Standard deviation for Guassian blur. This is ignored if using the
    # median blur filter. If less than 0, then this is automatically 
    # calculated
    gaus_sdev = unsharp_params[3]

    # Don't apply a sharp filter in these cases
    if amount <= 0:
        return img
    elif radius < 1:
        return img

    # Calculate the parameters for the Gaussian kernel
    k_size = (radius, radius) # Radius (i.e., kernel size)

    if gaus_sdev < 0:
        gaus_sdev = 0.3*((radius- 1)*0.5 - 1) + 0.8 # Standard deviation

    # Perform a blur filter
    if fltr_type:
        img_blur = cv.medianBlur(img, radius)
    else:
        img_blur = cv.GaussianBlur(img, k_size, gaus_sdev)

    img_sharp = cv.addWeighted(img, 1.0+amount, img_blur, -amount, 0)

    return img_sharp


def laplacian_sharp(img_in, sharp_params_in):
    """
    Sharpens an image using the edges detected by Laplacian.  The
    sharpening effect is achieved by subtracting the Laplacian image
    from the original (see below), 
        sharpened = original - (laplacian) X amount

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

    [sharp_params_in]: A list of parameters needed to perform the
        edge enhancement effect. Details of each parameter in 
        sharp_params_in is given below,

            [cur_amount, blur_k_size, lap_k_size, fltr_type, 
            blur_std_dev]

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

    ---- RETURNED ---- 
    [img_sharp]: Returns the resultant image after performing the
        image processing procedures. img_sharp is in the same format as
        img_in.
        
    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function.
    """
    
    # ---- Start Local Copies ----
    img = img_in.copy()
    sharp_params = sharp_params_in
    # ---- End Local Copies ----

    # As a percentage, this is a weight for adding images together. A 
    # typical value would be between 50% and 150%. Larger amount increases
    # the sharpening effect
    amount = sharp_params[0]/100.0 # Convert out of a percentage

    # Related to kernel size of the Gaussian kernel. This affects the edge
    # thickness of the sharpened image
    blur_radius = sharp_params[1] # In pixels
    if blur_radius % 2 == 0: # Must be odd for the Gaussian blur
        blur_radius += -1

    # Radius of the kernel used to calculate the derivatives (Laplacian)
    lap_radius = sharp_params[2]
    if lap_radius % 2 == 0: 
        lap_radius += -1

    # Affects which blur filter is used. 0 does a Gaussian blur, while 1
    # does a median blur, which is more edge-preserving.
    fltr_type = sharp_params[3]

    # Standard deviation for Guassian blur. This is ignored if using the
    # median blur filter. If less than 0, this is automatically calculated
    gaus_sdev = sharp_params[4]

    # Don't apply a sharp filter in these cases
    if amount <= 0:
        return img
    elif (lap_radius < 1) or (blur_radius < 1):
        return img

    # Calculate the parameters for the Gaussian kernel
    blur_k_size = (blur_radius, blur_radius) # Radius (i.e., kernel size)

    if gaus_sdev < 0:
        gaus_sdev = 0.3*((blur_radius- 1)*0.5 - 1) + 0.8 # Standard deviation

    # Perform a blur filter
    if fltr_type:
        img_blur = cv.medianBlur(img, blur_radius)
    else:
        img_blur = cv.GaussianBlur(img, blur_k_size, gaus_sdev)

    # Perform the Laplacian filter to find the edges based on local gradients
    img_lap_64 = cv.Laplacian(img_blur, cv.CV_64F, ksize=lap_radius)

    img_sharp = cv.addWeighted(img, 1.0, img_lap_64, -amount, 0, 
        dtype=cv.CV_8U)

    return img_sharp


def canny_sharp(img_in, sharp_params_in):
    """
    Sharpens an image based on the Canny Edge Detection algorithm. The
    sharpening effect is achieved by subtracting the Canny-edges image
    from the original (see below), 
        sharpened = original - (canny) X amount
    Note, the Canny Edge Detection algorithm finds edges and marks them
    with discrete lines, so the resultant sharpened image will most
    likely not exhibit smooth gradients in contrast near the edges. 

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

    [sharp_params_in]: A list of parameters needed to perform the
        edge enhancement effect. Details of each parameter in 
        sharp_params_in is given below,

            [k_size, cur_amount, thresh1, thresh2]

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

    ---- RETURNED ---- 
    [img_sharp]: Returns the resultant image after performing the
        image processing procedures. img_sharp is in the same format as
        img_in.
        
    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function.
    """
    
    # ---- Start Local Copies ----
    img = img_in.copy()
    sharp_params = sharp_params_in
    # ---- End Local Copies ----

    k_size = sharp_params[0]
    if k_size % 2 == 0: 
        k_size += -1

    amount = sharp_params[1]/100.0

    threshold1 = sharp_params[2]

    threshold2 = sharp_params[3]

    img_edges = cv.Canny(img, threshold1, threshold2, apertureSize=k_size, 
        L2gradient=True)

    img_sharp = cv.addWeighted(img, 1.0, img_edges, -amount, 0.0)

    return img_sharp


def global_equalize(img_in):
    """
    Enhances the contrast of a grayscale image by equalizing the 
    histogram. This is a global method, so the same parameters are
    utilized everywhere. For more information, vist "https://docs.
    opencv.org/4.2.0/d5/daf/tutorial_py_histogram_equalization.html"

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

    ---- RETURNED ---- 
    [img_eq]: The equalized image is returned in the same data type and
        format at img_in.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function.
    """

    # ---- Start Local Copies ----
    img = img_in.copy()
    # ---- End Local Copies ----

    img_eq = cv.equalizeHist(img)
    return img_eq


def adaptive_equalize(img_in, eq_params_in):
    """
    Enhances the contrast of a grayscale image by locally equalizing 
    the histogram. This is a locally adaptive method. For more 
    information, vist "https://docs.opencv.org/4.2.0/d5/daf/
    tutorial_py_histogram_equalization.html"

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

    [eq_params_in]: A list of parameters necessary for the locally 
        adaptive equalization procedure. More details of each parameter
        in eq_params_in is given below,

            [blk_size, c_offset]
                
                blk_size: Size of a pixel neighborhood that is used to
                    calculate a threshold value for the current pixel.
                    Must be odd, and if zero, no thresholding is done.
                
                c_offset: A constant offset value applied to the mean
                    or weighted mean of the intensities


    ---- RETURNED ---- 
    [img_eq]: The equalized image is returned in the same data type and
        format at img_in.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function.
    """

    # ---- Start Local Copies ----
    img = img_in.copy()
    eq_params = eq_params_in
    # ---- End Local Copies ----

    # Threshold for contrast limiting
    clip_limit = eq_params[0]

    # Size of grid for histogram equalization. Input image will be divided
    # into equally sized rectangular tiles. tileGridSize defines the number of
    # tiles in row and column.
    grid_size = eq_params[1]

    if grid_size < 1:
        return img

    tile_size = (grid_size, grid_size) 

    eq_obj = cv.createCLAHE(clip_limit, tile_size)
    img_eq = eq_obj.apply(img)

    return img_eq


def invert_binary_image(img_in, quiet_in=False):
    """
    Inverts a binarized (otherwise known as a black & white) image.
    Pixels that were black with intensity 0 become white with intensity
    255, and vice-versa.

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a black and white image. It is
        assumed  that the image is already of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

    ---- RETURNED ---- 
    [img]: The inverted image is returned. It is in the same format and
        data type as img_in.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function.
    """

    # ---- Start Local Copies ----
    img = img_in.copy()
    quiet = quiet_in
    # ---- End Local Copies ----

    img = cv.bitwise_not(img)

    if not quiet:
        print(f"\nSuccessfully inverted the image intensities.")

    return img


def calc_rel_density(img_in, invert_in=False, quiet_in=False):
    """
    Calculates the relative density of a binarized image, also known
    as a black and white image. By default, this function determines 
    the ratio of non-zero ratio pixels to the total number of pixels.
    Since white pixels are of intensity 255, another way to say this is
    that this function calculates portion of white pixels to total 
    pixels. A result of 1.0 implies the entire image is composed of
    non-zero pixels (i.e., white), and conversely, a result of 0.0 
    implies the entire image is black. However, it should be noted that
    the ratio of black pixels to total pixels can also be calculated,
    if desired.

    ---- INPUT ARGUMENTS ---- 
    [img_in]: OpenCV's numpy array for a black and white image. It is
        assumed  that the image is already of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

    invert_in: A boolean that determines whether white pixels or black
        pixels are of interest. If set to False, the returned result
        is, (number of non-zero pixels)/(total number of pixels). If set
        to True, (number of zero pixels)/(total number of pixels).

    quiet_in: A boolean that determines if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing.

    ---- RETURNED ---- 
    rel_density: A float between 0.0 and 1.0 representing the relative
        density. Multiply by 100 to get a percentage.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. Comments printed
    to standard output may occur.
    """
    quiet = quiet_in

    # Calculate number of non-black pixels
    num_white = cv.countNonZero(img_in)
    num_pixels = float(img_in.size) # Number of pixels in image

    if not invert_in:
        rel_density = num_white/num_pixels

    else:
        rel_density = (num_pixels - num_white)/num_pixels

    if not quiet:
        print(f"\nRelative Density of Final Image (from 0 to 1): "\
            f"{rel_density}")

    return rel_density


def create_blank_image(height_in, width_in, num_channels_in, 
    init_value_in=(0, 0, 0)):
    """
    Initializes a blank image with either 1 channel (grayscale) or 3
    channels (BGR format). The number of channels is also commonly
    referred to as the image depth. Images will be created using only
    the 'uint8' data type.

    ---- INPUT ARGUMENTS ----
    height_in: An integer representing the number of pixels in the 
        vertical direction (i.e., number of rows)

    width_in: An integer representing the number of pixels in the 
        horizontal direction (i.e., number of columns)

    num_channels_in: An integer that is either 1 or 3. Use 1 to create
        a single-channel image, which will be grayscale. Use 3 to 
        create a multi-channel image, which will be color. Recall that
        OpenCV uses Blue-Green-Red (BGR) format, rather than the 
        popular RGB format.

    (init_value_in): A tuple containing three integers. The blank image
        will be initialized to the color denoted by init_value_in. 
        Values should range between 0 and 255. If the desired image
        contains one channel, then the first value in init_value_in
        will be used to initialize the grayscale value.

    ---- RETURNED ---- 
    [img_blank]: The initialized image will be returned. The image will
        be based on OpenCV's NumPy data structure for images. The image
        will be of type 'uint8'.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function.
    """

    # Create local copies
    num_rows = height_in
    num_cols = width_in
    img_depth = num_channels_in
    img_color = init_value_in

    # Should be either depth of 1 (grayscale) or 3 (BGR)
    if (img_depth != 1) and (img_depth != 3):

        if img_depth < 1:
            img_depth = 1

        elif img_depth > 3:
            img_depth = 3

        else:
            img_depth = 1

    # If depth of 1, just need one intensity value. Using the first entry
    if img_depth == 1:
        img_color1 = img_color[0] 
        img_blank = (np.zeros((num_rows, num_cols), np.uint8))
        img_blank[:,:] = img_color1

    else: # Initialize the color for the BGR image
        if len(img_color) != 3:
            img_color = (0, 0, 0)

        img_blank = (np.zeros((num_rows, num_cols, 3), np.uint8))
        img_blank[:,:] = img_color

    return img_blank




    
