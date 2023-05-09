# Import external dependencies
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Slider, RangeSlider
from matplotlib.widgets import Button, CheckButtons, RadioButtons
from skimage import restoration as rest
from skimage import segmentation as seg
from skimage import filters as filt
from skimage import morphology as morph
from skimage.util import img_as_ubyte, img_as_float


def interact_gaussian_blur(img_in):
    """
    Placeholder to be completed later
    """
    pass


def interact_unsharp_mask(img_in):
    """
    Sharpens an image using unsharp masking, implemented in SciKit-Image
    via skimage.filters.unsharp_mask(). This is an interactive function
    that enables the user to change the parameters of the filter and see
    the results, thanks to the "widgets" available in Matplotlib.

     ---- INPUT ARGUMENTS ---- 
    [img_in]: Numpy array for a grayscale image. It is assumed that the
        image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

     ---- RETURNED ---- 
    [[img_out], [fltr_params]]: Returns a list containing the resultant 
        image and the parameters used for the filter. img_out is a
        sharpened image in the same format as img_in. fltr_params is
        also a list, which contains the final parameters used during
        the interactive session. The first item is the string name of
        the filter that was used, in this case "unsharp_mask". For this
        function, the [fltr_params] list contains:
            
            ["unsharp_mask", radius_out, amount_out]

                radius_out: Radius of the kernel for the unsharp filter. 
                    If zero, then no filter was applied. Should be an
                    integer.

                amount_out: The sharpening details will be amplified 
                    with this factor, which can be a negative or 
                    positive float.

     ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output. 
    """

    img_0 = img_in.copy()

    # Initial values for the filter
    radius_0 = 2
    amount_0 = 1

    img_temp = filt.unsharp_mask(img_0, radius=radius_0, 
        amount=amount_0, channel_axis=None)
    img_temp = img_as_ubyte(img_temp)

    # Global variables (within this function) that will be returned. 
    # Initialize these to the starting values for now.
    radius_out = radius_0
    amount_out = amount_0
    img_out = img_temp

    # Create a figure
    fig_size = (9,9)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax.set_aspect('equal')

    # Show the image and save the "matplotlib.image.AxesImage"
    # object for updating the figure later.
    img_obj = ax.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    ax.set_xlabel("X Pixel Number")
    ax.set_ylabel("Y Pixel Number")

    plt.subplots_adjust(bottom=0.26)

    # Create new axes objects for each button/slider/text widget
    # 4-tuple of floats rect = [left, bottom, width, height]. A new axes 
    # is added with dimensions rect in normalized (0, 1) units using 
    # add_axes on the current figure.
    radius_ax = fig.add_axes([0.21, 0.11, 0.15, 0.06])
    radius_text_box = TextBox(ax=radius_ax, label='Radius  ', 
        initial=str(radius_0), textalignment='center')

    amount_ax = fig.add_axes([0.66, 0.11, 0.15, 0.06])
    amount_text_box = TextBox(ax=amount_ax, label='Amount  ', 
        initial=str(amount_0), textalignment='center')

    update_ax = plt.axes([0.25, 0.03, 0.5, 0.05])
    update_button = Button(update_ax, 'Update')

    # Update the figure anytime the 'update' button is clicked
    def unsharp_mask_update(event):
        nonlocal radius_out
        nonlocal amount_out
        nonlocal img_out

        # The GUI widgets are defined in a higher-level scope, so
        # they can be accessed directly within this interior function 
        radius_out = float(radius_text_box.text)
        amount_out = float(amount_text_box.text)

        img_temp = filt.unsharp_mask(img_0, radius=radius_out, 
            amount=amount_out, channel_axis=None)
        img_out = img_as_ubyte(img_temp)

        # Update the image
        img_obj.set(data=img_out)

    # Call the update function when the 'update' button is clicked
    update_button.on_clicked(unsharp_mask_update)

    plt.show()

    # Save final filter parameters
    fltr_params = ["unsharp_mask", radius_out, amount_out]

    return [img_out, fltr_params]


def interact_hysteresis_threshold(img_in):
    """
    Applies a threshold to an image using hysteresis thresholding, which
    considers the connectivity of features. This is implemented in
    SciKit-Image via skimage.filters.apply_hysteresis_threshold(). This
    is an interactive function that enables the user to change the
    parameters of the filter and see the results, thanks to
    the "widgets" available in Matplotlib.

     ---- INPUT ARGUMENTS ---- 
    [img_in]: Numpy array for a grayscale image. It is assumed that the
        image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

     ---- RETURNED ---- 
    [[img_out], [fltr_params]]: Returns a list containing the resultant 
        image and the parameters used for the filter. img_out is a
        segmented image in the same format as img_in. Values will be
        either black (i.e., 0) or white (i.e., 255). fltr_params is
        also a list, which contains the final parameters used during
        the interactive session. The first item is the string name of
        the filter that was used, in this case "hysteresis_threshold".
        For this function, the [fltr_params] list contains:
            
            ["hysteresis_threshold", low_val_out, high_val_out]

                low_val_out: Lower threshold intensity as an integer

                high_val_out: Upper threshold intensity as an integer.

     ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output. 
    """

    img_0 = img_in.copy()

    # Initial values for the filter
    thresh_arr = filt.threshold_multiotsu(img_0, classes=3, nbins=256)
    low_val_0 = thresh_arr[0]
    high_val_0 = thresh_arr[1]

    img_temp = filt.apply_hysteresis_threshold(img_0, low_val_0, 
        high_val_0)
    img_temp = img_as_ubyte(img_temp)

    # Global variables (within this function) that will be returned. 
    # Initialize these to the starting values for now.
    low_val_out = low_val_0
    high_val_out = high_val_0
    img_out = img_temp

    # Create a figure
    fig_size = (9,9)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax.set_aspect('equal')

    # Show the image and save the "matplotlib.image.AxesImage"
    # object for updating the figure later.
    img_obj = ax.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    ax.set_xlabel("X Pixel Number")
    ax.set_ylabel("Y Pixel Number")

    plt.subplots_adjust(bottom=0.26)

    # Create new axes objects for each button/slider/text widget
    # 4-tuple of floats rect = [left, bottom, width, height]. A new axes 
    # is added with dimensions rect in normalized (0, 1) units using 
    # add_axes on the current figure.
    low_val_ax = fig.add_axes([0.25, 0.12, 0.60, 0.07])
    low_val_slider = Slider(ax=low_val_ax, label='Lower Threshold', 
        valmin=0, valmax=255, valinit=low_val_0, valstep=1.0)

    high_val_ax = fig.add_axes([0.25, 0.04, 0.60, 0.07])
    high_val_slider = Slider(ax=high_val_ax, label='Higher Threshold', 
        valmin=0, valmax=255, valinit=high_val_0, valstep=1.0)

    # Update the figure anytime the 'update' button is clicked
    def hysteresis_threshod_update(event):
        nonlocal low_val_out
        nonlocal high_val_out
        nonlocal img_out

        low_val_out = np.round(low_val_slider.val)
        low_val_out = low_val_out.astype(np.uint16)

        high_val_out = np.round(high_val_slider.val)
        high_val_out = high_val_out.astype(np.uint16)

        img_temp = filt.apply_hysteresis_threshold(img_0, low_val_out, 
        high_val_out)
        img_out = img_as_ubyte(img_temp)

        # Update the image
        img_obj.set(data=img_out)

    # Call the update function when the 'update' button is clicked
    low_val_slider.on_changed(hysteresis_threshod_update)
    high_val_slider.on_changed(hysteresis_threshod_update)

    plt.show()

    # Save final filter parameters
    fltr_params = ["hysteresis_threshold", low_val_out, high_val_out]

    return [img_out, fltr_params]


def interact_hysteresis_threshold2(img_in):
    """
    This function is virtually the same as the one above named,
    interact_hysteresis_threshold(). However, different widgets are
    used for this function. Instead of sliders, the user can provide
    thresholding inputs directly into textboxes. Otherwise, refer to
    interact_hysteresis_threshold() above for details about the inputs
    and outputs of this function.
    """

    img_0 = img_in.copy()

    # Initial values for the filter
    thresh_arr = filt.threshold_multiotsu(img_0, classes=3, nbins=256)
    low_val_0 = thresh_arr[0]
    high_val_0 = thresh_arr[1]

    img_temp = filt.apply_hysteresis_threshold(img_0, low_val_0, 
        high_val_0)
    img_temp = img_as_ubyte(img_temp)

    # Global variables (within this function) that will be returned. 
    # Initialize these to the starting values for now.
    low_val_out = low_val_0
    high_val_out = high_val_0
    img_out = img_temp

    # Create a figure
    fig_size = (9,9)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax.set_aspect('equal')

    # Show the image and save the "matplotlib.image.AxesImage"
    # object for updating the figure later.
    img_obj = ax.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    ax.set_xlabel("X Pixel Number")
    ax.set_ylabel("Y Pixel Number")

    plt.subplots_adjust(bottom=0.26)

    # Create new axes objects for each button/slider/text widget
    # 4-tuple of floats rect = [left, bottom, width, height]. A new axes 
    # is added with dimensions rect in normalized (0, 1) units using 
    # add_axes on the current figure.
    low_val_ax = fig.add_axes([0.22, 0.11, 0.15, 0.06])
    low_val_text_box = TextBox(ax=low_val_ax, label='Lower Threshold  ', 
        initial=str(low_val_0), textalignment='center')

    high_val_ax = fig.add_axes([0.71, 0.11, 0.15, 0.06])
    high_val_text_box = TextBox(ax=high_val_ax, label='Higher Threshold  ', 
        initial=str(high_val_0), textalignment='center')

    update_ax = plt.axes([0.25, 0.03, 0.5, 0.05])
    update_button = Button(update_ax, 'Update')

    # Update the figure anytime the 'update' button is clicked
    def hysteresis_threshod_update2(event):
        nonlocal low_val_out
        nonlocal high_val_out
        nonlocal img_out

        low_val_out = int(low_val_text_box.text)
        high_val_out = int(high_val_text_box.text)

        img_temp = filt.apply_hysteresis_threshold(img_0, low_val_out, 
        high_val_out)
        img_out = img_as_ubyte(img_temp)

        # Update the image
        img_obj.set(data=img_out)

    # Call the update function when the 'update' button is clicked
    update_button.on_clicked(hysteresis_threshod_update2)

    plt.show()

    # Save final filter parameters
    fltr_params = ["hysteresis_threshold", low_val_out, high_val_out]

    return [img_out, fltr_params]


def interact_sato_tubeness(img_in):
    """
    Filter an image with the Sato tubeness filter. This filter can be
    used to detect continuous ridges, e.g. tubes, wrinkles, rivers.
    This type of filter is also referred to as a ridge operator. It can
    be used to calculate the fraction of the whole image containing
    such objects. This is implemented in SciKit-Image via
    skimage.filters.sato(). This is an interactive function that
    enables the user to change the parameters of the filter and see the
    results, thanks to the "widgets" available in Matplotlib.

     ---- INPUT ARGUMENTS ---- 
    [img_in]: Numpy array for a grayscale image. It is assumed that the
        image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

     ---- RETURNED ---- 
    [[img_out], [fltr_params]]: Returns a list containing the resultant 
        image and the parameters used for the filter. img_out is a
        filterd image in the same format as img_in. fltr_params is
        also a list, which contains the final parameters used during
        the interactive session. The first item is the string name of
        the filter that was used, in this case "sato_tubeness". For this
        function, the [fltr_params] list contains:
            
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

     ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output. 
    """
    
    # Local copies
    img_0 = img_in.copy()

    # Initial values for the filter
    thresh_arr = filt.threshold_multiotsu(img_0, classes=3, nbins=256)
    mask_val_0 = thresh_arr[0]
    sig_max_0 = 10
    blk_ridges_0 = False

    mask = (img_0 > mask_val_0)
    img_temp = filt.sato(img_0, sigmas=range(1, sig_max_0), 
        black_ridges=blk_ridges_0)*mask

    img_temp = img_temp/np.amax(img_temp)
    img_temp = img_as_ubyte(img_temp)

    # Global variables (within this function) that will be returned. 
    # Initialize these to the starting values for now.
    mask_val_out = mask_val_0
    sig_max_out = sig_max_0
    blk_ridges_out = blk_ridges_0
    img_out = img_temp
    
    # Create a figure
    fig_size = (9,9)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax.set_aspect('equal')

    # Show the image and save the "matplotlib.image.AxesImage"
    # object for updating the figure later.
    img_obj = ax.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    ax.set_xlabel("X Pixel Number")
    ax.set_ylabel("Y Pixel Number")

    plt.subplots_adjust(bottom=0.26)

    # Create new axes objects for each button/slider/text widget
    # 4-tuple of floats rect = [left, bottom, width, height]. A new axes 
    # is added with dimensions rect in normalized (0, 1) units using 
    # add_axes on the current figure.
    mask_val_ax = fig.add_axes([0.17, 0.11, 0.15, 0.06])
    mask_val_text_box = TextBox(ax=mask_val_ax, label='Mask Intensity  ', 
        initial=str(mask_val_0), textalignment='center')

    sig_max_ax = fig.add_axes([0.51, 0.11, 0.15, 0.06])
    sig_max_text_box = TextBox(ax=sig_max_ax, label='Sigma Max  ', 
        initial=str(sig_max_0), textalignment='center')

    blk_ridges_ax = fig.add_axes([0.77, 0.11, 0.2, 0.08])
    blk_ridges_button = CheckButtons(ax=blk_ridges_ax, 
        labels=['Black Ridges  '], actives=[False])

    update_ax = plt.axes([0.25, 0.03, 0.5, 0.05])
    update_button = Button(update_ax, 'Update')

    # Update the figure anytime the 'update' button is clicked
    def sato_tubeness_update(event):
        # Use the new Python keyword 'nonlocal' to gain access and 
        # update these variables from within this scope.
        nonlocal mask_val_out
        nonlocal sig_max_out
        nonlocal blk_ridges_out
        nonlocal img_out

        # The GUI widgets are defined in a higher-level scope, so
        # they can be accessed directly within this interior function 
        mask_val_out = int(mask_val_text_box.text)
        sig_max_out = int(sig_max_text_box.text)
        blk_ridges_out = (blk_ridges_button.get_status())[0]

        mask = (img_0 > mask_val_out)
        img_temp = filt.sato(img_0, sigmas=range(1, sig_max_out), 
            black_ridges=blk_ridges_out)*mask

        img_temp = img_temp/np.amax(img_temp)
        img_out = img_as_ubyte(img_temp)

        # Update the image
        img_obj.set(data=img_out)

    # Call the update function when the 'update' button is clicked
    update_button.on_clicked(sato_tubeness_update)

    plt.show()

    # Save final filter parameters
    fltr_params = ["sato_tubeness", mask_val_out, sig_max_out, 
        blk_ridges_out]

    return [img_out, fltr_params]


def interact_tv_denoise(img_in):
    """
    Perform total-variation denoising on an image. The principle of
    total variation denoising is to minimize the total variation of the
    image, which can be roughly described as the integral of the norm
    of the image gradient. Total variation denoising tends to
    produce “cartoon-like” images, that is, piecewise-constant images.
    This is implemented in SciKit-Image via 
    skimage.restoration.denoise_tv_chambolle(). This is an interactive 
    function that enables the user to change the parameters of the 
    filter and see the results, thanks to the "widgets" available in 
    Matplotlib.

     ---- INPUT ARGUMENTS ---- 
    [img_in]: Numpy array for a grayscale image. It is assumed that the
        image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

     ---- RETURNED ---- 
    [[img_out], [fltr_params]]: Returns a list containing the resultant 
        image and the parameters used for the filter. img_out is a
        filterd image in the same format as img_in. fltr_params is
        also a list, which contains the final parameters used during
        the interactive session. The first item is the string name of
        the filter that was used, in this case "tv_chambolle". For this
        function, the [fltr_params] list contains:
            
            ["tv_chambolle", weight_out, eps_out, n_iter_max_out]

                weight_out: Denoising weight. The greater weight, the 
                    more denoising (at the expense of fidelity)

                eps_out: Relative difference of the value of the cost
                    function that determines the stop criterion. See the
                    Skimage documentation for additional details.

                n_iter_max_out: Maximal number of iterations used for 
                    the optimization.

     ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output. 
    """

    img_0 = img_in.copy()

    # Initial values for the filter
    weight_0 = 0.1
    eps_0 = 0.0002
    n_iter_max_0 = 200

    img_temp = rest.denoise_tv_chambolle(img_0, weight=weight_0,
        eps=eps_0, n_iter_max=n_iter_max_0, multichannel=False, 
        channel_axis=None)

    img_temp = img_as_ubyte(img_temp)

    # Global variables (within this function) that will be returned. 
    # Initialize these to the starting values for now.
    weight_out = weight_0
    eps_out = eps_0
    n_iter_max_out = n_iter_max_0
    img_out = img_temp

    # Create a figure
    fig_size = (9,9)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax.set_aspect('equal')

    # Show the image and save the "matplotlib.image.AxesImage"
    # object for updating the figure later.
    img_obj = ax.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    ax.set_xlabel("X Pixel Number")
    ax.set_ylabel("Y Pixel Number")

    plt.subplots_adjust(bottom=0.26)

    # Create new axes objects for each button/slider/text widget
    # 4-tuple of floats rect = [left, bottom, width, height]. A new axes 
    # is added with dimensions rect in normalized (0, 1) units using 
    # add_axes on the current figure.
    weight_ax = fig.add_axes([0.1, 0.11, 0.15, 0.06])
    weight_text_box = TextBox(ax=weight_ax, label='Weight  ', 
        initial=str(weight_0), textalignment='center')

    eps_ax = fig.add_axes([0.4, 0.11, 0.15, 0.06])
    eps_text_box = TextBox(ax=eps_ax, label='EPS  ', initial=str(eps_0),
        textalignment='center')

    n_iter_ax = fig.add_axes([0.8, 0.11, 0.15, 0.06])
    n_iter_text_box = TextBox(ax=n_iter_ax, label='Num. Iterations  ', 
        initial=str(n_iter_max_0), textalignment='center')

    update_ax = plt.axes([0.25, 0.03, 0.5, 0.05])
    update_button = Button(update_ax, 'Update')

    # Update the figure anytime the 'update' button is clicked
    def tv_denoise_update(event):
        # Use the new Python keyword 'nonlocal' to gain access and 
        # update these variables from within this scope.
        nonlocal weight_out
        nonlocal eps_out
        nonlocal n_iter_max_out
        nonlocal img_out

        # The GUI widgets are defined in a higher-level scope, so
        # they can be accessed directly within this interior function 
        weight_out = float(weight_text_box.text)
        eps_out = float(eps_text_box.text)
        n_iter_max_out = int(n_iter_text_box.text)

        img_out = rest.denoise_tv_chambolle(img_0, weight=weight_out,
        eps=eps_out, n_iter_max=n_iter_max_out, multichannel=False, 
        channel_axis=None)

        img_out = img_as_ubyte(img_out)

        # Update the image
        img_obj.set(data=img_out)

    # Call the update function when the 'update' button is clicked
    update_button.on_clicked(tv_denoise_update)

    plt.show()

    # Save final filter parameters
    fltr_params = ["tv_chambolle", weight_out, eps_out, n_iter_max_out]

    return [img_out, fltr_params]


def interact_nl_means_denoise(img_in):
    """
    Perform non-local means denoising on an image. The non-local means
    algorithm is well suited for denoising images with specific
    textures. The principle of the algorithm is to average the value of
    a given pixel with values of other pixels in a limited
    neighbourhood, provided that the patches centered on the other
    pixels are similar enough to the patch centered on the pixel of
    interest. This is implemented in SciKit-Image via
    skimage.restoration.denoise_nl_means(). Note, for the Skimage
    implementation, fast_mode=True has been chosen here. Moreover, the
    the standard deviation of the Gaussian noise is being estimated
    beforehand using skimage.restoration.estimate_sigma(). Additionally,
    the image input should be UINT8 but will be converted to a float
    during image processing. Although the returned image will still be
    consistent (as a UINT8), the input parameters for this function are
    actually based on the converted image as a float. Specifically,
    this only affects h_out described below. This is an interactive
    function that enables the user to change the parameters of the
    filter and see the results, thanks to the "widgets" available in
    Matplotlib.

     ---- INPUT ARGUMENTS ---- 
    [img_in]: Numpy array for a grayscale image. It is assumed that the
        image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel.

     ---- RETURNED ---- 
    [[img_out], [fltr_params]]: Returns a list containing the resultant 
        image and the parameters used for the filter. img_out is a
        filterd image in the same format as img_in. fltr_params is
        also a list, which contains the final parameters used during
        the interactive session. The first item is the string name of
        the filter that was used, in this case "nl_means". For this
        function, the [fltr_params] list contains:
            
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

     ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output. 
    """

    img_0 = img_in.copy()

    # Initial estimate of the standard deviation in the noise assuming 
    # a Gaussian distribution
    sig_noise = rest.estimate_sigma(img_as_float(img_0), 
        average_sigmas=True, channel_axis=None)

    # Initial filter parameters
    h_0 = 0.8*sig_noise
    patch_size_0 = 5
    patch_dist_0 = 7

    img_temp = rest.denoise_nl_means(img_0, h=h_0, sigma=sig_noise, 
        fast_mode=True, patch_size=patch_size_0, 
        patch_distance=patch_dist_0, channel_axis=None)

    img_temp = img_as_ubyte(img_temp)

    # Global variables (within this function) that will be returned. 
    # Initialize these to the starting values for now.
    h_out = h_0
    patch_size_out = patch_size_0
    patch_dist_out = patch_dist_0
    img_out = img_temp

    # Create a figure
    fig_size = (9,9)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax.set_aspect('equal')

    # Show the image and save the "matplotlib.image.AxesImage"
    # object for updating the figure later.
    img_obj = ax.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    ax.set_xlabel("X Pixel Number")
    ax.set_ylabel("Y Pixel Number")

    plt.subplots_adjust(bottom=0.26)

    # Create new axes objects for each button/slider/text widget
    # 4-tuple of floats rect = [left, bottom, width, height]. A new axes 
    # is added with dimensions rect in normalized (0, 1) units using 
    # add_axes on the current figure.
    h_ax = fig.add_axes([0.17, 0.11, 0.15, 0.06])
    h_text_box = TextBox(ax=h_ax, label='Cut-Off Range  ', 
        initial=f"{h_0:.4f}", textalignment='center')

    p_size_ax = fig.add_axes([0.47, 0.11, 0.15, 0.06])
    p_size_text_box = TextBox(ax=p_size_ax, label='Patch Size  ', 
        initial=str(patch_size_out), textalignment='center')

    p_dist_ax = fig.add_axes([0.82, 0.11, 0.15, 0.06])
    p_dist_text_box = TextBox(ax=p_dist_ax, label='Search Distance  ', 
        initial=str(patch_dist_out), textalignment='center')

    update_ax = plt.axes([0.25, 0.03, 0.5, 0.05])
    update_button = Button(update_ax, 'Update')

    # Update the figure anytime the 'update' button is clicked
    def nl_means_denoise_update(event):
        # Use the new Python keyword 'nonlocal' to gain access and 
        # update these variables from within this scope.
        nonlocal h_out
        nonlocal patch_size_out
        nonlocal patch_dist_out
        nonlocal img_out

        # The GUI widgets are defined in a higher-level scope, so
        # they can be accessed directly within this interior function 
        h_out = float(h_text_box.text)
        patch_size_out = int(p_size_text_box.text)
        patch_dist_out = int(p_dist_text_box.text)

        img_out = rest.denoise_nl_means(img_0, h=h_out, 
        sigma=sig_noise, fast_mode=True, patch_size=patch_size_out, 
        patch_distance=patch_dist_out, channel_axis=None)

        img_out = img_as_ubyte(img_out)

        # Update the image
        img_obj.set(data=img_out)

    # Call the update function when the 'update' button is clicked
    update_button.on_clicked(nl_means_denoise_update)

    plt.show()

    # Save final filter parameters
    fltr_params = ["nl_means", h_out, patch_size_out, patch_dist_out]

    return [img_out, fltr_params]


def interact_binary_morph(img_in):
    """
    Perform a morphological filter operation on a binary
    (i.e., segmented) image. More specifically, apply either an
    erosion, dilation, "opening", or "closing" with the option of
    choosing different kernel shapes and sizes. The implementations of
    these operations are based on the Skimage library. This is an
    interactive function that enables the user to change the parameters
    of the filter and see the results, thanks to the "widgets"
    available in Matplotlib.

     ---- INPUT ARGUMENTS ---- 
    [img_in]: Numpy array for a binarized image. It is assumed that the
        image is already of type uint8. The array should thus be 2D,
        where each value represents the intensity for each
        corresponding pixel.

     ---- RETURNED ---- 
    [[img_out], [fltr_params]]: Returns a list containing the resultant 
        image and the parameters used for the filter. img_out is a
        filterd image in the same format as img_in. fltr_params is
        also a list, which contains the final parameters used during
        the interactive session. For this function, the [fltr_params]
        list contains:
            
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

     ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. It does pop-up
    a new window that visualizes the provided image. Strings are printed
    to standard output. 
    """

    img_0 = img_in.copy()

    # Determine what type of morphological operation to perform
    # 0: binary_closing
    # 1: binary_opening
    # 2: binary_dilation
    # 3: binary_erosion
    operation_type_0 = 0

    # Determine what type of 2D neighborhood to use.
    # 0: square (which corresponds to a cube in 3D)
    # 1: disk (which corresponds to a ball in 3D)
    # 2: diamond (which corresponds to a octahedron in 3D)
    footprint_type_0 = 1

    # Radius of the footprint neighborhood (in pixels)
    n_radius_0 = 1

    disk_footprint = morph.disk(n_radius_0)
    img_temp_0 = morph.binary_closing(img_0, footprint=disk_footprint)
    img_temp_0 = img_as_ubyte(img_temp_0)

    # Global variables (within this function) that will be returned. 
    # Initialize these to the starting values for now.
    operation_type_out = operation_type_0
    footprint_type_out = footprint_type_0
    n_radius_out = n_radius_0
    img_out = img_temp_0.copy()

    # Create a figure
    fig_size = (9,9)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax.set_aspect('equal')

    # Show the image and save the "matplotlib.image.AxesImage"
    # object for updating the figure later.
    img_obj = ax.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    ax.set_xlabel("X Pixel Number")
    ax.set_ylabel("Y Pixel Number")

    plt.subplots_adjust(bottom=0.32)

    # Create new axes objects for each button/slider/text widget
    # 4-tuple of floats rect = [left, bottom, width, height]. A new axes 
    # is added with dimensions rect in normalized (0, 1) units using 
    # add_axes on the current figure.
    operation_ax = fig.add_axes([0.10, 0.03, 0.15, 0.20])
    operation_button = RadioButtons(ax=operation_ax, 
        labels=['Closing', 'Opening', 'Dilation', 'Erosion'],
        active=operation_type_0)

    footprint_ax = fig.add_axes([0.36, 0.03, 0.15, 0.20])
    footprint_button = RadioButtons(ax=footprint_ax, 
        labels=['Square', 'Disk', 'Diamond'],
        active=footprint_type_0)

    radius_ax = fig.add_axes([0.75, 0.15, 0.15, 0.06])
    radius_text_box = TextBox(ax=radius_ax, label='Kernel Radius  ', 
        initial=str(n_radius_0), textalignment='center')

    update_ax = plt.axes([0.65, 0.03, 0.25, 0.06])
    update_button = Button(update_ax, 'Update')

    def binary_morph_update(event):
        nonlocal operation_type_out
        nonlocal footprint_type_out
        nonlocal n_radius_out
        nonlocal img_out

        # Retrieve the current values
        operation_type_str = operation_button.value_selected
        footprint_type_str = footprint_button.value_selected
        n_radius_out = int(radius_text_box.text)

        # Convert string names to the internal flag codes
        if footprint_type_str == 'Square':
            footprint_type_out = 0
            temp_width = 2*n_radius_out + 1
            temp_footprint = morph.square(temp_width)

        elif footprint_type_str == 'Disk':
            footprint_type_out = 1
            temp_footprint = morph.disk(n_radius_out)

        elif footprint_type_str == 'Diamond':
            footprint_type_out = 2
            temp_footprint = morph.diamond(n_radius_out)
            
        else:
            print("\nWarning: Footprint type could not be retrieved."\
                "\nDefaulting to Disk.")
            footprint_type_out = 1
            temp_footprint = morph.disk(n_radius_out)

        # Convert string names to the internal flag codes
        if operation_type_str == 'Closing':
            operation_type_out = 0
            img_temp = morph.binary_closing(img_0, footprint=temp_footprint)

        elif operation_type_str == 'Opening':
            operation_type_out = 1
            img_temp = morph.binary_opening(img_0, footprint=temp_footprint)

        elif operation_type_str == 'Dilation':
            operation_type_out = 2
            img_temp = morph.binary_dilation(img_0, footprint=temp_footprint)

        elif operation_type_str == 'Erosion':
            operation_type_out = 3
            img_temp = morph.binary_erosion(img_0, footprint=temp_footprint)

        else:
            print("\nWarning: Operation type could not be retrieved."\
                "\nDefaulting to Binary Closing.")
            operation_type_out = 1
            img_temp = morph.binary_closing(img_0, footprint=temp_footprint)

        img_out = img_as_ubyte(img_temp)

        # Update the image
        img_obj.set(data=img_out)

    # Call the update function when the 'update' button is clicked
    update_button.on_clicked(binary_morph_update)

    plt.show()

    # Save final filter parameters
    fltr_params = [operation_type_out, footprint_type_out,
        n_radius_out]

    return [img_out, fltr_params]

