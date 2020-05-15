# OPENCV-PIPELINE
USE_OPENCV = True                        # boolean flag for whether images are preprocessed using opencv
CV_STORE = True  # whether to store raw-image or pre-processed image (TRUE == store cv-processed image)
CV_PREPROCESSOR_TYPE = 'combined_with_gray'     # describes type of pre-processing
                                      # (canny | segmentation | combined | combined_with_gray)
CV_TARGET_IMAGE_DEPTH = 3
CV_STORE_INF = False # whether to store "inference-input" during ai-mode for testing
CV_COLOR_MODE = 'indoor'  # mode for color segmentation (indoor | outdoor)

# OPENCV COLOR SEGMENTATION SETTING - INDOOR
# WHITE
CV_WHITE_LOWER_IN = [140, 140, 140]
CV_WHITE_UPPER_IN = [255, 255, 255]
# YELLOW
CV_YELLOW_LOWER_IN = [25, 110, 30]
CV_YELLOW_UPPER_IN = [70, 160, 70]

# OPENCV COLOR SEGMENTATION SETTING - OUTDOOR
# WHITE
CV_WHITE_LOWER_OUT = [170, 170, 170]
CV_WHITE_UPPER_OUT = [255, 255, 255]
# YELLOW
CV_YELLOW_LOWER_OUT = [5, 100, 70]
CV_YELLOW_UPPER_OUT = [20, 180, 170]

# OPENCV CANNY SETTING
CV_CANNY_MIN = 50
CV_CANNY_MAX = 125
CV_CANNY_APPLY_HOUGH = True
CV_HOUGH_MIN_VOTES = 20
CV_HOUGH_LINE_THICKNESS = 5
CV_MIN_LINE_LENGTH = 5
CV_MAX_LINE_GAP = 10

# OPENCV ROI SELECTION SETTING
CV_MAKE_COORDINATE_UPPER_LIMIT = 3 / 4
CV_ROI_TYPE = 'crop'                # type of roi-operation (crop | mask | None)
CV_ROI_Y_UPPER_EDGE = 65

# OPENCV GAUSSIAN BLUR setting
CV_GAUSSIAN_KERNEL = (5, 5)
CV_GAUSSIAN_SIGMA = 0.0  # might be unnecessary at all, if 0 is being used

#Cropping 
#Numbers go from 0 to 1
CROP_TOP = 0.5
CROP_BOTTOM = 1.0
CROP_LEFT = 0.5
CROP_RIGHT = 1.0

#Image to use for auto-cropping
import os
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
CROPPING_FILENAME = os.path.join(CURRENT_PATH, "../../dataset/tub_1_20-03-07/40_cam-image_array_.jpg")

# Saving data
SAVE_CROP_DATA = False  
SAVE_ORIGINAL_DATA = False