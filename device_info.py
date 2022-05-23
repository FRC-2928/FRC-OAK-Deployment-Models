#!/usr/bin/env python3

import depthai as dai
import cv2

pipeline = dai.Pipeline()

# Create nodes, configure them and link them together
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")

# Linking
camRgb.preview.link(xoutRgb.input)

# Upload the pipeline to the device
with dai.Device(pipeline) as device:
    # Print Myriad X Id (MxID), USB speed, and available cameras on the device
    print('MxId:',device.getDeviceInfo().getMxId())
    print('USB speed:',device.getUsbSpeed())
    print('Connected cameras:',device.getConnectedCameras())

    # Input queue, to send message from the host to the device (you can receive the message on the device with XLinkIn)
    #   input_q = device.getInputQueue("input_name", maxSize=4, blocking=False)

    # Output queue, to receive message on the host from the device (you can send the message on the device with XLinkOut)
    #   output_q = device.getOutputQueue("output_name", maxSize=4, blocking=False)
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        # Get a message that came from the queue
        inPreview = previewQueue.get() # Or output_q.tryGet() for non-blocking
        frame = inPreview.getCvFrame()
        cv2.imshow("rgb", frame)

        # Send a message to the device
        # cfg = dai.ImageManipConfig()
        # input_q.send(cfg)

        if cv2.waitKey(1) == ord('q'):
            break