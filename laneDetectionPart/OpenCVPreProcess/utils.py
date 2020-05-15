import cv2 as cv
import numpy as np


def roi(cfg, height, width):
    """
    Creates the rectangular region of interest
    
    Args:
        cfg: {obj} -- representing specified config-parameters in config.py / myconfig.py
        height: {int} -- height of rectangle
        width: {int} -- width of rectangle
        
    Returns:
        np.array -- coordinates of each point of the rectangle
    """
    return np.array([[(0, height), (width, height),
                      (width, cfg.CV_ROI_Y_UPPER_EDGE), (0, cfg.CV_ROI_Y_UPPER_EDGE)]])


def make_coordinates(image, line_parameters, cfg):
    """
    Converts the given slope and intercept of a line into pixel points
    
    Args:
        image: {np.array} -- original image
        line_parameters: {np.array} -- array containing (slope,intercept) of line
        cfg: {obj} -- representing specified config-parameters in config.py / myconfig.py
        
    Returns:
        np.array -- pixel points on the line specified by the given slope and intercept
    """
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * cfg.CV_MAKE_COORDINATES_UPPER_LIMIT)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines, cfg):
    """
    Takes in coordinates of lines, gets the slope and intercept, decide if it is a right or left lane line
    depending on slope, and return an averaged coordinates for left and right lane line
    
    Args:
        image: {np.array} -- original image
        lines: {np.array} -- contains points on a line
        cfg: {obj} -- representing specified config-parameters in config.py / myconfig.py
    
    Returns:
        np.array -- contains the averaged coordinates for a left and right lane line
    """
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average, cfg)
    right_line = make_coordinates(image, right_fit_average, cfg)

    return np.array([left_line, right_line])


def bgr_to_hls(image):
    '''Convert BRG image to HLS color space
    
    Arguments:
        image {np array} -- original image in BGR color space
    
    Returns:
        np array -- converted image in HLS color space
    '''
    return cv.cvtColor(image, cv.COLOR_BGR2HLS)


def bgr_to_hsv(image):
    '''Convert bgr image to hsv color space

    Arguments:
        image {np array} -- original BGR image
    
    Returns:
        np array -- image in BGR color space
    '''
    return cv.cvtColor(image, cv.COLOR_BGR2HSV)


def rgb_to_hls(image):
    """
    Convert RGB image to HLS color space

    Arguments:
        image {np array} -- original RGB image

    Returns:
        nd array -- converted image in HLS color space
    """

    return cv.cvtColor(image, cv.COLOR_RGB2HLS)


def rgb_to_hsv(image):
    """
    Convert image to hsv color space

    Arguments:
        image {np array} -- original RGB image
    Returns:
        nd array -- converted image in HSV color space
    """

    return cv.cvtColor(image, cv.COLOR_RGB2HSV)


def create_color_mask(img, col_lower, col_upper):
    """
    Threshold the image to specific color ranges
    
    Args:
        img: {np.array} -- HLS color space image to be used
        col_lower: {int} -- number value of low color threshold
        col_upper: {int} -- number value of upper color threshold
        
    Returns:
        mask: image with pixel values in the range specified by col_lower and col_upper
    """
    mask = cv.inRange(img, col_lower, col_upper)
    return mask


def create_line_image(image, lines, cfg):
    """
    Draws lines identified in the image onto a black background
    
    Args:
        image: {np.array} -- image that the lines come from
        lines: {np.array} -- coordinates of the lines (edges) in image
        cfg: {obj} -- representing specified config-parameters in config.py / myconfig.py
    
    Returns:
        line_image: {np.array} -- black background image with lines representing the edges of the original image drawn on it
    """
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), cfg.CV_HOUGH_LINE_THICKNESS)

    return line_image


def visualize_edges(img, edges):
    """
    Show the edges in the image
    
    Args:
        img: {np.array} -- original image
        edges: {?} -- edges in the image

    Returns:
        edges overlayed with the image
    """
    # cv.imshow('edges', edges)

    combo_image = cv.addWeighted(img, 0.8, cv.cvtColor(edges, cv.COLOR_GRAY2BGR), 1, 1)
    cv.imshow('edges-combo', combo_image)


def visualize_segmentation(img, mask, mask_2=None):
    """
    Shows the HSV yellow and orange lanes
    
    Args:
        image: {np.array} -- image to be shown
        mask: {?} -- 
        mask_2: {?}
    
    Returns:
        2 images showing the segmented regions and the segmented region overlayed with the image
    """
    if mask_2 is not None:
        cv.imshow('segmented_lane_white', cv.bitwise_and(img, img, mask=mask))
        cv.imshow('segmented_lane_orange', cv.bitwise_and(img, img, mask=mask_2))
    else:
        # cv.imshow('segmented_region', mask)
        cv.imshow('segmented_lane_combo', cv.bitwise_and(img, img, mask=mask))


def apply_canny(image, cfg):
    """
    Perform canny on image to detect edge with high contrast
    You can tone the cv2.Canny(blur, 50,150) parameter. 50 is low
    threshold and 150 is high threshold

    :param image: {np array} -- original image
    :param cfg: obj -- {obj} representing specified config-parameters in config.py / myconfig.py
    :return: canny: {np array} -- image after canny function
    """

    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, cfg.CV_GAUSSIAN_KERNEL, cfg.CV_GAUSSIAN_SIGMA)
    canny = cv.Canny(blur, cfg.CV_CANNY_MIN, cfg.CV_CANNY_MAX)
    return canny


def region_of_interest(image, cfg):
    """
    Only displays the region of interest in the image. Everything not in the region is blacked out.
    
    Args:
        image: {np.array} -- image you want to get region of interest on
        cfg: {obj} representing specified config-parameters in config.py / myconfig.py
    
    Returns:
        masked_image: {np.array} -- image containing only everything from the original image inside the region of interest
    """
    height = image.shape[0]
    width = image.shape[1]

    polygons = roi(cfg, height, width)

    if len(image.shape) == 2:
        mask = np.zeros_like(image)
        cv.fillPoly(mask, polygons, 255)
        masked_image = cv.bitwise_and(image, mask)

    else:
        mask = np.zeros_like(image[:, :, 0])
        cv.fillPoly(mask, polygons, 255)
        channels = image.shape[2]
        masked_image = np.zeros_like(image)

        for c in range(channels):
            masked_image[:, :, c] = cv.bitwise_and(image[:, :, c], mask)

    return masked_image
