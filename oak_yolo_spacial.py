#!/usr/bin/env python3

import os
import json
import numpy as np
import threading
import argparse
import time
from time import sleep
from pathlib import Path
from pathlib import Path
import cv2
import depthai as dai

from wpi_helpers import ConfigParser, WPINetworkTables, ModelConfigParser, WPINetworkTables

'''
Spatial Tiny-yolo example
  Performs inference on RGB camera and retrieves spatial location 
  coordinates: x,y,z relative to the center of depth map.
  Detected objects are displayed to localhost:8091 
  
  The script uses the WPI Network Tables to send data back to the WPI program.
  Can be used for tiny-yolo-v3 or tiny-yolo-v4 networks  
'''

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with OpenVINO optimized '
            'YOLO model')
    parser = argparse.ArgumentParser(description=desc)
    # parser = add_camera_args(parser)
    parser.add_argument(
        '-g', '--gui', action='store_true',
        help='use desktop gui for display [False]')
    parser.set_defaults(gui=False)    
    parser.add_argument(
        '-t', '--conf_thresh', type=float, default=0.3,
        help='set the detection confidence threshold')
    parser.add_argument(
        '-m', '--model', type=str, required=True, default='custom',
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish|yolov4-p5]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    parser.add_argument(
        '-p', '--mjpeg_port', type=int, default=8080,
        help='MJPEG server port [8080]')    
    args = parser.parse_args()
    return args

def draw_boxes(detection, frame, label, color):
    # Denormalize bounding box
    x1 = int(detection.xmin * frame.shape[1])
    x2 = int(detection.xmax * frame.shape[1])
    y1 = int(detection.ymin * frame.shape[0])
    y2 = int(detection.ymax * frame.shape[0])

    x_coord = int(detection.spatialCoordinates.x)   
    y_coord = int(detection.spatialCoordinates.y)
    z_coord = int(detection.spatialCoordinates.z)
    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
    cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
    cv2.putText(frame, f"X: {x_coord} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
    cv2.putText(frame, f"Y: {y_coord} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
    cv2.putText(frame, f"Z: {z_coord} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
    return frame
           
def loop_and_detect(previewQueue, detectionNNQueue, depthQueue, 
                    xoutBoundingBoxDepthMappingQueue, labelMap, nt, cvSource):
    """Continuously capture images from camera and do object detection.

    # Arguments
      previewQueue: Image data stream 
      detectionNNQueue: Detection objects 
      depthQueue: Object depth information 
      xoutBoundingBoxDepthMappingQueue: Bounding boxes for objects 
      labelMap: Map of labelled classes
      nt: the WPI Network Tables.
      cvSource: The source going out to the mjpeg server
    """
    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)

    # Run detection loop
    while True:
        inPreview = previewQueue.get()
        inDet = detectionNNQueue.get()
        depth = depthQueue.get()

        frame = inPreview.getCvFrame()
        depthFrame = depth.getFrame() # depthFrame values are in millimeters

        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        detections = inDet.detections

        if len(detections) != 0:
            boundingBoxMapping = xoutBoundingBoxDepthMappingQueue.get()
            roiDatas = boundingBoxMapping.getConfigData()

            for roiData in roiDatas:
                roi = roiData.roi
                roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                topLeft = roi.topLeft()
                bottomRight = roi.bottomRight()
                xmin = int(topLeft.x)
                ymin = int(topLeft.y)
                xmax = int(bottomRight.x)
                ymax = int(bottomRight.y)

                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)


        # If the frame is available, draw bounding boxes on it and show the frame
        height = frame.shape[0]
        width  = frame.shape[1]
        for detection in detections:
            try:
                label = labelMap[detection.label]
            except:
                label = detection.label

            frame = draw_boxes(detection, frame, label, color)

            # Put data to Network Tables
            nt.put_spacial_data(detection, label, fps)

        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        
        if cvSource is False:
            # Display stream to desktop window
            cv2.imshow("depth", depthFrameColor)
            cv2.imshow("rgb", frame)
        else:               
            # Display stream to browser
            cvSource.putFrame(frame)   

        if cv2.waitKey(1) == ord('q'):
            break

# -------------------------------------------------------------------------
# Main Program Start
# -------------------------------------------------------------------------
def main(args, config_parser):
    
    # Get the model blob file
    if not os.path.isfile('%s.blob' % args.model):
        raise SystemExit('ERROR: file (%s.blob) not found!' % args.model)

    blob_file = f"{args.model}.blob"
    config_file = f"{args.model}-config.json"
    nnPath = str((Path(__file__).parent / Path(blob_file)).resolve().absolute())
    configPath = str((Path(__file__).parent / Path(config_file)).resolve().absolute())
    print(f"Running model at path {nnPath}")

    if not Path(nnPath).exists():
        print(f"No model found at path {nnPath}")

    ## Read the model configuration file
    print("Loading network settings")
    model_config = ModelConfigParser(configPath)
    print(model_config.labelMap)
    print("Classes:", model_config.classes)
    print("Confidence Threshold:", model_config.confidence_threshold)

    print("Connecting to Network Tables")
    hardware_type = "OAK-D Camera"
    nt = WPINetworkTables(config_parser.team, hardware_type, model_config.labelMap)

    syncNN = True

    # Configure and load the camera pipeline
    print("Loading camera and model")
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutNN = pipeline.create(dai.node.XLinkOut)
    xoutBoundingBoxDepthMapping = pipeline.create(dai.node.XLinkOut)
    xoutDepth = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")
    xoutNN.setStreamName("detections")
    xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
    xoutDepth.setStreamName("depth")

    # Properties
    frame_width = 416
    frame_height = 416
    camRgb.setPreviewSize(frame_width, frame_height)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # setting node configs
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    spatialDetectionNetwork.setBlobPath(nnPath)
    spatialDetectionNetwork.setConfidenceThreshold(model_config.confidence_threshold)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    # Yolo specific parameters
    spatialDetectionNetwork.setNumClasses(model_config.classes)
    spatialDetectionNetwork.setCoordinateSize(4)
    spatialDetectionNetwork.setAnchors(np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]))
    spatialDetectionNetwork.setAnchorMasks({ "side26": np.array([1,2,3]), "side13": np.array([3,4,5]) })
    spatialDetectionNetwork.setIouThreshold(0.5)

    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    camRgb.preview.link(spatialDetectionNetwork.input)
    if syncNN:
        spatialDetectionNetwork.passthrough.link(xoutRgb.input)
    else:
        camRgb.preview.link(xoutRgb.input)

    spatialDetectionNetwork.out.link(xoutNN.input)
    spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

    stereo.depth.link(spatialDetectionNetwork.inputDepth)
    spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

    # Connect to device and start pipeline
    print("Connecting to device and starting pipeline")
    with dai.Device(pipeline) as device:

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        xoutBoundingBoxDepthMappingQueue = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        # Run the inference loop
        if args.gui is True:
            print("Gui requested")
            try:
                loop_and_detect(previewQueue, detectionNNQueue, 
                                depthQueue, xoutBoundingBoxDepthMappingQueue, 
                                model_config.labelMap, nt, cvSource=False)
            except Exception as e:
                print(e)
            finally:
                print("Finished") 
        else:
            # Start the mjpeg server (default)
            import cscore as cs
            cvSource = cs.CvSource("cvsource", cs.VideoMode.PixelFormat.kMJPEG, 320, 240, 30)
            mjpeg_server = cs.MjpegServer("httpserver", args.mjpeg_port)
            mjpeg_server.setSource(cvSource)
            print('MJPEG server started on port', args.mjpeg_port)
            try:
                loop_and_detect(previewQueue, detectionNNQueue, 
                                depthQueue, xoutBoundingBoxDepthMappingQueue, 
                                model_config.labelMap, nt, cvSource=cvSource)
            except Exception as e:
                print(e)
            finally:
                print("Finished")         


if __name__ == '__main__':
    print("Running oak_yolo_spacial_wpi.py")
    args = parse_args()

    # Load the FRC configuration file
    config_parser = ConfigParser()

    main(args, config_parser)        
