#!/usr/bin/env python3

import depthai as dai
import pandas as pd
import os
import cv2 # Must be imported otherwise cscore import hangs
from datetime import datetime

global imgList, steeringList
countFolder = 0
count = 0
imgList = []
steeringList = []

#GET CURRENT DIRECTORY PATH
myDirectory = os.path.join(os.getcwd(), 'DataCollected')
# print(myDirectory)

# CREATE A NEW FOLDER BASED ON THE PREVIOUS FOLDER COUNT
while os.path.exists(os.path.join(myDirectory,f'IMG{str(countFolder)}')):
        countFolder += 1
newPath = myDirectory +"/IMG"+str(countFolder)
os.makedirs(newPath)

# SAVE IMAGES IN THE FOLDER
def saveData(img,steering):
    global imgList, steeringList
    now = datetime.now()
    timestamp = str(datetime.timestamp(now)).replace('.', '')
    #print("timestamp =", timestamp)
    fileName = os.path.join(newPath,f'Image_{timestamp}.jpg')
    cv2.imwrite(fileName, img)
    imgList.append(fileName)
    steeringList.append(steering)

# SAVE LOG FILE WHEN THE SESSION ENDS
def saveLog():
    global imgList, steeringList
    rawData = {'Image': imgList,
                'Steering': steeringList}
    df = pd.DataFrame(rawData)
    df.to_csv(os.path.join(myDirectory,f'log_{str(countFolder)}.csv'), index=False, header=False)
    print('Log Saved')
    print('Total Images: ',len(imgList))

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutPreview = pipeline.create(dai.node.XLinkOut)
xoutPreview.setStreamName('video')

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Linking
camRgb.preview.link(xoutPreview.input)

# Start the mjpeg server (default)
try:
    import cscore as cs
    mjpeg_port = 8080
    cvSource = cs.CvSource("cvsource", cs.VideoMode.PixelFormat.kMJPEG, 320, 240, 30)
    mjpeg_server = cs.MjpegServer("httpserver", mjpeg_port)
    mjpeg_server.setSource(cvSource)
    print('MJPEG server started on port', mjpeg_port)
except Exception as e:
    cvSource = False

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Print Myriad X Id (MxID), USB speed, and available cameras on the device
    print('MxId:',device.getDeviceInfo().getMxId())
    print('USB speed:',device.getUsbSpeed())
    print('Connected cameras:',device.getConnectedCameras())

    # Output queue will be used to get the encoded data from the output defined above
    previewQueue = device.getOutputQueue(name="video", maxSize=4, blocking=True)
    # previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    print("Press Ctrl+C to stop encoding...")
    try:
        while True:
            # Save stream
            previewFrame = previewQueue.get()
            frame = previewFrame.getFrame()
            saveData(frame, 0)

            # Display stream
            if cvSource is False:
                # Display stream to desktop window
                cv2.imshow("rgb", frame)
            else:               
                # Display stream to browser
                cvSource.putFrame(frame)   

    except KeyboardInterrupt:
        # Keyboard interrupt (Ctrl + C) detected
        pass

    saveLog()

    print("To view the encoded data, convert the stream file (.h265) into a video file (.mp4) using a command below:")
    print("ffmpeg -framerate 30 -i video.h265 -c copy video.mp4")