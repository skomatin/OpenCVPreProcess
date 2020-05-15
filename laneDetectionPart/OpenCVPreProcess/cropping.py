import cv2
import numpy as np

class Crop(object):
	
	def __init__(self, cfg):
		self.cfg = cfg

	def run(self, image):
		try:
			h, w, c = image.shape
			cropped_image = image[int(self.cfg.CROP_TOP*h): int(self.cfg.CROP_BOTTOM*h), int(self.cfg.CROP_LEFT*w): int(self.cfg.CROP_RIGHT*w)]
			return cropped_image
		except:
			pass
 
class CropSelect():

	def __init__(self):
		import openCVconfig as cfg
		#Variables for the mouse click callback function
		self.cropping = False
		self.x_start, self.y_start, self.x_end, self.y_end = 0, 0, 0, 0
		self.image = cv2.imread(cfg.CROPPING_FILENAME)
		self.h, self.w, self.c = self.image.shape

	def mouse_crop(self, event, x, y, flags, param):
		# grab references to the global variables
		# global x_start, y_start, x_end, y_end, cropping
	
		# if the left mouse button was DOWN, start RECORDING
		# (x, y) coordinates and indicate that cropping is being
		if event == cv2.EVENT_LBUTTONDOWN:
			self.x_start, self.y_start, self.x_end, self.y_end = x, y, x, y
			self.cropping = True
	
		# Mouse is Moving
		elif event == cv2.EVENT_MOUSEMOVE:
			if self.cropping == True:
				self.x_end, self.y_end = x, y
	
		# if the left mouse button was released
		elif event == cv2.EVENT_LBUTTONUP:
			# record the ending (x, y) coordinates
			self.x_end, self.y_end = x, y
			self.cropping = False # cropping is finished

			TOP = min(self.y_start/self.h, 1.0)
			BOTTOM = min(self.y_end/self.h, 1.0)
			LEFT = min(self.x_start/self.w, 1.0)
			RIGHT = min(self.x_end/self.w, 1.0)
			if TOP < 0 or BOTTOM < 0 or LEFT < 0 or RIGHT < 0:
				print("Error Selecting Region:\nPlease Select the region within the image")
			elif LEFT > RIGHT or TOP > BOTTOM:
				print("Error Selecting Region:\nPlease Select the region from top left to bottom right")
			else:
				print("Cropping configurations: \nTOP = {:.2f}\nBOTTOM = {:.2f}\nLEFT = {:.2f}\nRIGHT = {:.2f}".format(TOP, BOTTOM, LEFT, RIGHT))
				print("\nPlease update the openCVconfig file with these values")
				print("-------------------------------------------------------")

	def start(self):
		cv2.namedWindow("image")
		cv2.setMouseCallback("image", self.mouse_crop)
		
		while True:
		
			self.i = self.image.copy()
		
			if not self.cropping:
				cv2.imshow("image", self.image)
		
			elif self.cropping:
				cv2.rectangle(self.i, (self.x_start, self.y_start), (self.x_end, self.y_end), (255, 0, 0), 2)
				cv2.imshow("image", self.i)
		
			if cv2.waitKey(1) == 27:
				break
 
		# close all open windows
		cv2.destroyAllWindows()

if __name__ == '__main__':
	cropSelect = CropSelect()
	cropSelect.start()
	