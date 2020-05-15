# import myconfig as cfg
from donkeycar.vehicle import Vehicle
from cvImg import CvCam, CvImageDisplay, CvImg

# from cvPreProcessSai.cvPreProcess import OpenCVPreProcess
import OpenCVPreProcess.openCVconfig as cfg
import OpenCVPreProcess.OpenCVPreProcess as OP
from OpenCVPreProcess.cropping import Crop

V = Vehicle()

# cam = CvCam()
# V.add(cam, outputs = ["camera/image"], threaded = True)

img = CvImg()
V.add(img, outputs = ["cam/image_array"])

# preProcess = OpenCVPreProcess()
# V.add(preProcess, inputs = ["camera/image"], outputs = ["camera/image"])

preProcess = OP.OpenCvPreProcessor(cfg)
V.add(preProcess, inputs = ["cam/image_array"], outputs = ["cam/image_array_preprocess"])

cropPart = Crop(cfg)
V.add(cropPart, inputs = ["cam/image_array_preprocess"], outputs = ["cam/image_array_cropped"])

disp = CvImageDisplay(False)
V.add(disp, inputs = ["cam/image_array_cropped"])

V.start()

