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
        '-n', '--no_network_tables', action='store_true',
        help='use WPI Network Tables [True]')
    parser.set_defaults(no_network_tables=False)   
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

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def displayFrame(name, detection, frame):
    color = (255, 0, 0)
    bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
    # cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
    cv2.putText(frame, f"{int(detection.confidence * 100)}%", cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

    # Show the frame
    cv2.imshow(name, frame)
           
def loop_and_detect(previewQueue, detectionNNQueue, networkTables, cvSource):
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
        inRgb = previewQueue.get()
        inDet = detectionNNQueue.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        # If the frame is available, draw bounding boxes on it and show the frame
        if frame is not None:

            # If the frame is available, draw bounding boxes on it and show the frame
            for detection in detections:

                print(detection)

                frame = displayFrame(detection, frame, color)

                # Put data to Network Tables
                if networkTables:
                    networkTables.put_spacial_data(detection, fps)

        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        
        if cvSource is False:
            # Display stream to desktop window
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
    # print("Loading network settings")
    # model_config = ModelConfigParser(configPath)
    # print(model_config.labelMap)
    # print("Classes:", model_config.classes)
    # print("Confidence Threshold:", model_config.confidence_threshold)

    print("Connecting to Network Tables")
    hardware_type = "OAK-D Camera"
    if args.no_network_tables == False:
        print("Using Network Tables")
        networkTables = WPINetworkTables(config_parser.team, hardware_type)
    else:
        print("No Network Tables requested")
        networkTables = False    

    syncNN = True

    # Configure and load the camera pipeline
    print("Loading camera and model")
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    nn = pipeline.create(dai.node.NeuralNetwork)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutNN = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")
    xoutNN.setStreamName("detections")


    # Properties
    frame_width = 200
    frame_height = 200
    camRgb.setPreviewSize(frame_width, frame_height)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # Setting node configs
    nn.setBlobPath(args.nnPath)
    nn.setConfidenceThreshold(0.5)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)

    # Linking
    camRgb.preview.link(nn.input)
    if syncNN:
        nn.passthrough.link(xoutRgb.input)
    else:
        camRgb.preview.link(xoutRgb.input)
    nn.out.link(xoutNN.input)

    # Connect to device and start pipeline
    print("Connecting to device and starting pipeline")
    with dai.Device(pipeline) as device:

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        
        # Run the inference loop
        if args.gui is True:
            print("Gui requested")
            try:
                loop_and_detect(previewQueue, detectionNNQueue, networkTables, cvSource=False)
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
                loop_and_detect(previewQueue, detectionNNQueue, networkTables, cvSource=cvSource)
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
