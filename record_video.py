#!/usr/bin/env python3

import av
import depthai as dai
from fractions import Fraction
import time
import sys
import argparse

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with OpenVINO optimized '
            'YOLO model')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-c', '--codec', type=str, default="mjpeg",
        help='codec can be either h264, h265, or mjpeg')

    args = parser.parse_args()
    return args    


def get_encoder_profile(codec):
    if codec == "h264": return dai.VideoEncoderProperties.Profile.H264_MAIN
    elif codec == "mjpeg": return dai.VideoEncoderProperties.Profile.MJPEG
    else: return dai.VideoEncoderProperties.Profile.H265_MAIN

# -------------------------------------------------------------------------
# Main Program Start
# -------------------------------------------------------------------------
def main(args):

    # codec = "hevc" # H265 by default
    # if 2 <= len(sys.argv):
    #     codec = sys.argv[1].lower()
    #     if codec == "h265": codec = "hevc"

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and output
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)

    # Properties
    print(f"codec {args.codec}")
    videoEnc = pipeline.create(dai.node.VideoEncoder)
    videoEnc.setDefaultProfilePreset(30, get_encoder_profile(args.codec))
    # videoEnc.setLossless(True) # Lossless MJPEG, video players usually don't support it
    camRgb.video.link(videoEnc.input)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName('enc')
    videoEnc.bitstream.link(xout.input)

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        print(f"App starting streaming {get_encoder_profile(args.codec).name} encoded frames into file video.mp4")

        # Output queue will be used to get the encoded data from the output defined above
        q = device.getOutputQueue(name="enc", maxSize=30, blocking=True)

        output_container = av.open('video.mp4', 'w')
        stream = output_container.add_stream(args.codec, rate=30)
        stream.time_base = Fraction(1, 1000 * 1000) # Microseconds

        if args.codec == "mjpeg":
            # We need to set pixel format for MJEPG, for H264/H265 it's yuv420p by default
            stream.pix_fmt = "yuvj420p"

        start = time.time()
        try:
            while True:
                data = q.get().getData() # np.array
                packet = av.Packet(data) # Create new packet with byte array

                # Set frame timestamp
                packet.pts = int((time.time() - start) * 1000 * 1000)

                output_container.mux_one(packet) # Mux the Packet into container

        except KeyboardInterrupt:
            # Keyboard interrupt (Ctrl + C) detected
            pass

        output_container.close()

if __name__ == '__main__':
    print("Running record_video.py")
    args = parse_args()

    main(args)    