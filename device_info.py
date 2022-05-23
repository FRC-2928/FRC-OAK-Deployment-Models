#!/usr/bin/env python3

import cv2 # Must be imported otherwise cscore import hangs
import depthai as dai
import cscore as cs

pipeline = dai.Pipeline()

# Create nodes, configure them and link them together
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")

# Linking
camRgb.preview.link(xoutRgb.input)

# Start the mjpeg server (default)
mjpeg_port = 8080
cvSource = cs.CvSource("cvsource", cs.VideoMode.PixelFormat.kMJPEG, 320, 240, 30)
mjpeg_server = cs.MjpegServer("httpserver", mjpeg_port)
mjpeg_server.setSource(cvSource)
print('MJPEG server started on port', mjpeg_port)

# Upload the pipeline to the device
with dai.Device(pipeline) as device:
    # Print Myriad X Id (MxID), USB speed, and available cameras on the device
    print('MxId:',device.getDeviceInfo().getMxId())
    print('USB speed:',device.getUsbSpeed())
    print('Connected cameras:',device.getConnectedCameras())

    # Output queue, to receive message on the host from the device (you can send the message on the device with XLinkOut)
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    try:
        while True:
            # Get a message that came from the queue
            inPreview = previewQueue.get() # Or output_q.tryGet() for non-blocking
            frame = inPreview.getCvFrame()

            # Display stream to browser
            cvSource.putFrame(frame)   

    except KeyboardInterrupt:
        # Keyboard interrupt (Ctrl + C) detected
        pass  