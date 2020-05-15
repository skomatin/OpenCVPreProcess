import cv2
import time
import os

class CvCam(object):
    '''
    Purpose: A Donkeycar part to get frames from the camera and return it
    '''

    def __init__(self, iCam=0):
        '''
        Purpose: To initialize all variables required for the part
        '''

        self.cap = cv2.VideoCapture(iCam)   #Camera feed capture object
        self.frame = None                   #Contains the latest frame from the camera
        self.running = True                 #Indicates whether the thread is running

    def poll(self):
        '''
        Purpose: To read the latest frame from the webcam and store it in self.frame
        '''
        _, self.frame = self.cap.read()

    def update(self):
        '''
        Purpose: to run a thread simulaneously with the main thread 
        '''
        while (self.running):
            self.poll()                     #Keep collecting frames as long as thread is running

    def run_threaded(self):
        '''
        Purpose: Return the latest frame when called by the vehicle with threaded = True
        '''
        return self.frame

    def run(self):
        '''
        Purpose: Return the latest frame when called by the vehicle
        '''
        self.poll()
        return self.frame
    
    def shutdown(self):
        '''
        Purpose: End frame capture and set running to false
        '''
        self.running = False                #Stop the thread
        time.sleep(0.2)                     #Provide buffer time to allow thread to shutdown before cap.release() gets called
        self.cap.release()

class CvImg(object):
    
    def __init__(self, path):
        self.frame = None
        self.running = True
        self.fileName = 1
        self.path = path

    def poll(self):
        filePath = os.path.join(self.path, f'tub_3_20-03-07/{self.fileName}_cam-image_array_.jpg')
        self.frame = cv2.imread(filePath)
        self.fileName += 1

    def update(self):
        while (self.running):
            self.poll()

    def run_threaded(self):
        return self.frame

    def run(self):
        self.poll()
        return self.frame
    
    def shutdown(self):
        self.running = False
        time.sleep(0.2)

class CvImageDisplay(object):
    '''
    Purpose: To display a given image
    '''

    def __init__(self, save=False):
        self.save = save

    def run(self, image):
        try:
            cv2.imshow('frame', image)
            if self.save:
                # print("Saving Image")
                cv2.imwrite('../savedImg.png', image)
            cv2.waitKey(1)
        except:
            pass

    def shutdown(self):
        cv2.destroyAllWindows()