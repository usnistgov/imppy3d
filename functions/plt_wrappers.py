# Import external dependencies
import numpy as np
import matplotlib.pyplot as plt

from skimage.util import img_as_ubyte


def create_bw_fig(img_in, show_histogram=False):
    """
    Creates a black and white figure. The inputs are an 8-bit image,
    defined as a 2D Numpy array, and a boolean. When the boolean is set
    to True, a histogram of the image will also be shown. 
    """

    img = img_in.copy()
    #img = img_as_ubyte(img)

    if show_histogram:
        # (Width, Height) in inches
        fig_size = (12,6)

        # Show the histogram to the right of the grayscale image
        fig1, ax1 = plt.subplots(1, 2)

        # Big enough, and still fit in modern monitors
        fig1.set_size_inches(fig_size[0], fig_size[1])

        # Include vmin & vmax, else imshow() automatically normalizes from 0 to 1
        ax1[0].set_aspect('equal')
        ax1[0].imshow(img, cmap='gray', vmin=0, vmax=255)
        #ax1[0].set_title("Image")
        ax1[0].set_xlabel("X Pixel Number")
        ax1[0].set_ylabel("Y Pixel Number")

        ax1[1].hist(img.ravel(),256,[0,256])
        #ax1[1].set_title("Histogram")
        ax1[1].set_xlabel("Grayscale Intensity")
        ax1[1].set_ylabel("Counts") 


    else:
        # (Width, Height) in inches
        fig_size = (6,6)

        # Using subplot to make a single image since it conveniently returns
        # the image figure and axes objects
        fig1, ax1 = plt.subplots(1, 1)

        # Big enough, and still fit in modern monitors
        fig1.set_size_inches(fig_size[0], fig_size[1])

        # Include vmin & vmax, else imshow() automatically normalizes from 0 to 1
        ax1.set_aspect('equal')
        ax1.imshow(img, cmap='gray', vmin=0, vmax=255)
        #ax1.set_title("Image")
        ax1.set_xlabel("X Pixel Number")
        ax1.set_ylabel("Y Pixel Number") 


    # This function returns the figure object (as fig1) and the axes object
    # (as ax1). To actually show these images in the calling script, type the
    # following:

    """
        # Makes the subplots fit better in the figure's canvas. 
        plt.tight_layout() 

        plt.show()
    """

    # Note, this function is blocking: the script will not continue until
    # the figures are closed. To show the figures in a non-blocking manner,
    # use the following code:

    """
        # Makes the subplots fit better in the figure's canvas. 
        plt.tight_layout() 

        plt.show(block=False)
        plt.pause(1.0) # Pause for one second while the figure loads
    """

    # In this case, it is up to the user to ensure that the figure reference
    # is closed before the script terminates. To do so, use the plt.close()
    # function, i.e.  plt.close('all'). 

    return [fig1, ax1]


def create_2_bw_figs(img1_in, img2_in):
    """
    Creates 2 black and white figures, side-by-side. The inputs are two
    8-bit images, each defined as a 2D Numpy array. 
    """

    img_1 = img1_in.copy()
    img_2 = img2_in.copy()
    #img_1 = img_as_ubyte(img_1)
    #img_2 = img_as_ubyte(img_2)

    # (Width, Height) in inches
    fig_size = (12,6)

    # Link the axes of both images
    fig1, ax1 = plt.subplots(1, 2, sharex='row', sharey='row')

    # Big enough, and still fit in modern monitors
    fig1.set_size_inches(fig_size[0], fig_size[1])

    # Include vmin & vmax, else imshow() automatically normalizes from 0 to 1
    ax1[0].set_aspect('equal')
    ax1[0].imshow(img_1, cmap='gray', vmin=0, vmax=255)
    #ax1[0].set_title("Image 1")
    ax1[0].set_xlabel("X Pixel Number")
    ax1[0].set_ylabel("Y Pixel Number")

    ax1[1].set_aspect('equal')
    ax1[1].imshow(img_2, cmap='gray', vmin=0, vmax=255)
    #ax1[1].set_title("Image 2")
    ax1[1].set_xlabel("X Pixel Number")
    ax1[1].set_ylabel("Y Pixel Number")

    # This function returns the figure object (as fig1) and the axes object
    # (as ax1). To actually show these images in the calling script, type the
    # following:

    """
        # Makes the subplots fit better in the figure's canvas. 
        plt.tight_layout() 

        plt.show()
    """

    # Note, this function is blocking: the script will not continue until
    # the figures are closed. To show the figures in a non-blocking manner,
    # use the following code:

    """
        # Makes the subplots fit better in the figure's canvas. 
        plt.tight_layout() 

        plt.show(block=False)
        plt.pause(1.0) # Pause for one second while the figure loads
    """

    # In this case, it is up to the user to ensure that the figure reference
    # is closed before the script terminates. To do so, use the plt.close()
    # function, i.e.  plt.close('all').

    return [fig1, ax1]



