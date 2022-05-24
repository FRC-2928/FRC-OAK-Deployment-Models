#!/usr/bin/env python3

import depthai as dai
import img_helpers as img
import cv2 # Must be imported otherwise cscore import hangs

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutPreview = pipeline.create(dai.node.XLinkOut)
xoutPreview.setStreamName('preview')

# Properties
camRgb.setPreviewSize(300, 300)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(True)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

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
    # previewQueue = device.getOutputQueue(name="preview", maxSize=4, blocking=False)
    previewQueue = device.getOutputQueue('preview')

    print("Press Ctrl+C to stop encoding...")
    try:
        while True:
            # Save stream
            previewFrame = previewQueue.get()
            img.saveData(previewFrame.getFrame(), 0)

            # Display stream
            if cvSource is False:
                # Display stream to desktop window
                cv2.imshow("rgb", previewFrame.getCvFrame())
            else:               
                # Display stream to browser
                cvSource.putFrame(previewFrame.getFrame())   

    except KeyboardInterrupt:
        # Keyboard interrupt (Ctrl + C) detected
        pass

    print("Saving log file")    
    img.saveLog()