#!/usr/bin/env python3

import depthai as dai
import cv2 # Must be imported otherwise cscore import hangs

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and output
camRgb = pipeline.create(dai.node.ColorCamera)
videoEnc = pipeline.create(dai.node.VideoEncoder)
xout = pipeline.create(dai.node.XLinkOut)
xoutPreview = pipeline.create(dai.node.XLinkOut)

xout.setStreamName('h265')
xoutPreview.setStreamName('preview')

# Properties
# frame_width = 416
# frame_height = 416
# camRgb.setPreviewSize(frame_width, frame_height)

camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
# camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Create encoder consuming the frames and encoding them using H.264 / H.265 encoding
videoEnc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.H265_MAIN)

# Linking
camRgb.video.link(videoEnc.input)
videoEnc.bitstream.link(xout.input)
camRgb.preview.link(xoutPreview.input)

# Start the mjpeg server (default)  This parts not working
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
    previewQueue = device.getOutputQueue(name="preview")
    h265Queue = device.getOutputQueue(name="h265", maxSize=30, blocking=True)
    # previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    # The .h265 file is a raw stream file (not playable yet)
    with open('video.h265', 'wb') as videoFile:
        print("Press Ctrl+C to stop encoding...")
        try:
            while True:
                h265Packet = h265Queue.get()  # Blocking call, will wait until a new data has arrived
                h265Packet.getData().tofile(videoFile)  # Appends the packet data to the opened file

                # Display stream
                previewFrame = previewQueue.get()
                frame = previewFrame.getCvFrame()
                if cvSource is False:
                    # Display stream to desktop window
                    cv2.imshow("rgb", frame)
                else:               
                    # Display stream to browser
                    cvSource.putFrame(frame)   
        except KeyboardInterrupt:
            # Keyboard interrupt (Ctrl + C) detected
            pass

    print("To view the encoded data, convert the stream file (.h265) into a video file (.mp4) using a command below:")
    print("ffmpeg -framerate 30 -i video.h265 -c copy video.mp4")