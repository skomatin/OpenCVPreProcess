import cv2 as cv
import numpy as np
# from utils import *
from utils import *
from donkeycar.utils import img_crop

class Dummy:
    '''
    Rreturned output is the same as input
    '''
    def __init__(self):
        pass

    def run(self, image_arr):
        return image_arr

class _OpenCvPreProcessor:
    '''
    Abstract class of image processor
    Implement frame work of shutdown and run function as required by 
    Donkey car framework.
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        pass

    def shutdown(self):
        return

    def run(self, image_arr):
        raise NotImplementedError


class OpenCvPreProcessor:
    """
    preprocessor-manager defining kind of pre-processing based 
    on a string-argument loads corresponding class in init
    """

    def __init__(self, cfg):
        self.cfg = cfg

        if cfg.CV_PREPROCESSOR_TYPE == 'canny':
            assert cfg.CV_TARGET_IMAGE_DEPTH == 1
            self.processor = OpenCvCanny(cfg)

        elif cfg.CV_PREPROCESSOR_TYPE == 'segmentation':
            assert cfg.CV_TARGET_IMAGE_DEPTH == 1
            self.processor = OpenCvColorSegmentation(cfg)

        elif cfg.CV_PREPROCESSOR_TYPE == 'combined':
            assert cfg.CV_TARGET_IMAGE_DEPTH == 3
            self.processor = OpenCvCannyAndSegmentation(cfg)

        elif cfg.CV_PREPROCESSOR_TYPE == 'combined_with_gray':
            assert cfg.CV_TARGET_IMAGE_DEPTH == 3
            self.processor = OpenCvCannySegmentationAndGray(cfg)

        else:
            raise NotImplementedError

    def run(self, image_arr):
        if image_arr is None:
            return None

        processed_image = self.processor.run(image_arr)
        return processed_image

    def shutdown(self):
        self.processor.shutdown()


class OpenCvCannyAndSegmentation(_OpenCvPreProcessor):
    '''
    Convert original RGB image into three channel images with
    canny,white lines, and yellow lines
    
    Extends:
        _OpenCvPreProcessor
    '''
    def __init__(self, cfg):
        super().__init__(cfg)
        self.canny = OpenCvCanny(cfg)
        self.segmentation = OpenCvColorSegmentation(cfg, separate_masks=True)

    def run(self, image_arr):
        edges = self.canny.run(image_arr)
        white_mask, orange_mask = self.segmentation.run(image_arr)

        return cv.merge([edges, white_mask, orange_mask])


class OpenCvCannySegmentationAndGray(_OpenCvPreProcessor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.canny = OpenCvCanny(cfg)
        self.segmentation = OpenCvColorSegmentation(cfg, separate_masks=False)

    def run(self, image_arr):
        edges = self.canny.run(image_arr)
        lane_mask = self.segmentation.run(image_arr)
        gray = cv.cvtColor(image_arr, cv.COLOR_RGB2GRAY)

        return cv.merge([gray, edges, lane_mask])


class OpenCvCanny(_OpenCvPreProcessor):
    '''
    Detect edges of image using canny filter
 
    Extends:
        _OpenCvPreProcessor
    '''
    def __init__(self, cfg):
        super().__init__(cfg)

    def run(self, image_arr):
        canny_edges = apply_canny(image_arr, self.cfg)

        if self.cfg.CV_ROI_TYPE == 'mask':
            edges_roi = region_of_interest(canny_edges, self.cfg)
        else:
            edges_roi = canny_edges

        if self.cfg.CV_CANNY_APPLY_HOUGH:
            lines = cv.HoughLinesP(edges_roi, 1, np.pi / 180, self.cfg.CV_HOUGH_MIN_VOTES, np.array([]),
                                   minLineLength=self.cfg.CV_MIN_LINE_LENGTH,
                                   maxLineGap=self.cfg.CV_MAX_LINE_GAP)
            edges_roi = create_line_image(np.zeros_like(edges_roi), lines, self.cfg)

        return edges_roi


class OpenCvColorSegmentation(_OpenCvPreProcessor):
    '''
    Convert image to HSL color space and then segment white and
    yellow color.

    Extends:
        _OpenCvPreProcessor
    '''
    def __init__(self, cfg, separate_masks=False):
        super().__init__(cfg)
        self.separate_masks = separate_masks

        if cfg.CV_COLOR_MODE == 'indoor':
            self.white_lower = np.array(cfg.CV_WHITE_LOWER_IN)
            self.white_upper = np.array(cfg.CV_WHITE_UPPER_IN)

            self.orange_lower = np.array(cfg.CV_YELLOW_LOWER_IN)
            self.orange_upper = np.array(cfg.CV_YELLOW_UPPER_IN)

        elif cfg.CV_COLOR_MODE == 'outdoor':
            self.white_lower = np.array(cfg.CV_WHITE_LOWER_OUT)
            self.white_upper = np.array(cfg.CV_WHITE_UPPER_OUT)

            self.orange_lower = np.array(cfg.CV_YELLOW_LOWER_OUT)
            self.orange_upper = np.array(cfg.CV_YELLOW_UPPER_OUT)

    def run(self, image_arr):
        conv_img = rgb_to_hls(image_arr)

        if self.cfg.CV_ROI_TYPE == 'mask':
            conv_img = region_of_interest(conv_img, self.cfg)

        white_mask = create_color_mask(image_arr, self.white_lower, self.white_upper)
        orange_mask = create_color_mask(conv_img, self.orange_lower, self.orange_upper)

        if self.separate_masks:
            return_mask = white_mask, orange_mask
        else:
            return_mask = cv.bitwise_or(white_mask, orange_mask)

        return return_mask