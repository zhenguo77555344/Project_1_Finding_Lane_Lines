import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import os
from IPython.display import HTML

import math


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_lines_new(img, lines, color=[255, 0, 0], thickness=5):
    i = 0
    lineLeftX = []
    lineLeftY = []
    lineRightX = []
    lineRightY = []
    kLeft = []
    kRight = []
    for line in lines:
        i = i + 1
        for x1, y1, x2, y2 in line:
            # print("k=",(y2-y1)/(x2-x1))
            k = (y2 - y1) / (x2 - x1)
            if k < 0:
                lineLeftX.append(x1)
                lineLeftX.append(x2)
                lineLeftY.append(y1)
                lineLeftY.append(y2)
                kLeft.append(k)
            else:
                lineRightX.append(x1)
                lineRightX.append(x2)
                lineRightY.append(y1)
                lineRightY.append(y2)
                kRight.append(k)
    #  print("kLeft=",kLeft)
    #  print("kRight=",kRight)

    kLeftAvg = np.mean(kLeft)
    kRightAvg = np.mean(kRight)

    b0 = lineLeftY[0] - kLeftAvg * lineLeftX[0]
    b1 = lineRightY[0] - kRightAvg * lineRightX[0]

    startLeft = math.ceil((img.shape[0] - b0) / kLeftAvg)
    startRight = math.ceil((img.shape[0] - b1) / kRightAvg)

    #cv2.line(img, (min(lineLeftX), max(lineLeftY)), (max(lineLeftX), min(lineLeftY)), color, thickness)
    #cv2.line(img, (max(lineRightX), max(lineRightY)), (min(lineRightX), min(lineRightY)), color, thickness)
    cv2.line(img, (startLeft, img.shape[0]), (max(lineLeftX), min(lineLeftY)), color, thickness)
    cv2.line(img, (startRight, img.shape[0]), (min(lineRightX), min(lineRightY)), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img, lines)
    draw_lines_new(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def process_image_with_plot(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image where lines are drawn on lanes)

    ##Step 1: grayscale
    gray = grayscale(image)

    ##Step 2: Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    ##Step 3: Canny detection
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    ##Step 4: Keeps the region of interest
    imshape = image.shape
    vertices = np.array([[(190, imshape[0]), (350, 350), (590, 325), (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    ##Step 5: Hough transform

    # Make a blank the same size as our image to draw on
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = 1 * np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 20  # 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 35  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on
    lines_hough = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

    ##Step 6:Draw lines on a blank image
    lines_edges = weighted_img(lines_hough, image, α=0.8, β=1., γ=0.)
    result = lines_edges

    # figure plot
    plt.figure("Lane Detection")
    plt.subplot(2, 4, 1)
    plt.title("Original Picture")
    plt.imshow(image)

    plt.subplot(2, 4, 2)
    plt.title("Grayscale Picture")
    plt.imshow(gray, cmap='gray')

    plt.subplot(2, 4, 3)
    plt.title("Gaussian smoothing")
    plt.imshow(blur_gray, cmap='gray')

    plt.subplot(2, 4, 4)
    plt.title("Canny Detection")
    plt.imshow(edges, cmap='gray')

    plt.subplot(2, 4, 5)
    plt.imshow(masked_edges, cmap='gray')
    plt.title("Region of Interest")

    plt.subplot(2, 4, 6)
    plt.imshow(lines_hough)
    plt.title("Hough Transform")

    plt.subplot(2, 4, 7)
    plt.imshow(lines_edges)
    plt.title("Weighted Image")



    plt.show()

    return result


def process_image(image):
    ##Step 1: grayscale
    gray = grayscale(image)

    ##Step 2: Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    ##Step 3: Canny detection
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    ##Step 4: Keeps the region of interest
    imshape = image.shape
    vertices = np.array([[(190, imshape[0]), (350, 350), (590, 325), (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    ##Step 5: Hough transform

    # Make a blank the same size as our image to draw on
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = 1 * np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 20  # 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 35  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on
    lines_hough = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

    ##Step 6:Draw lines on a blank image
    lines_edges = weighted_img(lines_hough, image, α=0.8, β=1., γ=0.)
    result = lines_edges

    return result




def main():
    image = mping.imread('test_images/solidWhiteCurve.jpg')
    #image = mping.imread('test_images/solidWhiteRight.jpg')
    #image = mping.imread('test_images/solidYellowCurve.jpg')
    #image = mping.imread('test_images/solidYellowCurve2.jpg')
    #image = mping.imread('test_images/solidYellowLeft.jpg')
    #image = mping.imread('test_images/whiteCarLaneSwitch.jpg')
    #print('This image is: ', type(image), 'with dimensions: ', image.shape)

    process_image_with_plot(image)
'''
    white_output = 'test_videos_output/solidWhiteRight.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0, 5)
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
'''

if __name__ =="__main__":
    main()






