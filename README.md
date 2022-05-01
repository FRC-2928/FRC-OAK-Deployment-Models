# FRC-OAK-Deployment-Models
This repository stores *Deep Learning* models and scripts that can be deployed on the Raspberry Pi or Jetson Nano. This example uses an attached [Luxonis OAK](https://shop.luxonis.com/products/1098obcenclosure) camera.  The OAK camera has an imbedded TPU the runs on the [Intel OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html). The OAK camera's TPU runs the inference model, taking the processing off of the host processor.  

Before deploying these models the Raspberry Pi must have the WPILibPi Romi image installed.  If running on a Jetson Nano that must be installed with Jetpack 4.6.  The main files are included in this repository are as follows:

- `oak_yolo_spacial.py`  This is the default script that runs inference on a Yolo model and outputs detected objects with a label, bounding boxes and their X, Y, Z coordinates from the camera.  The script will display its output in a Web browser at `<server IP address:8080` and also places all of the data into the *WPILib* Network Tables.

- `oak_yolo.py`  This is a lighter version of the above script that only collects the label and bounding box data.

- `oak_yolo_spacial_gui.py`  This can be used to display camera stream output to the Jetson Nano desktop gui or any other device that has a gui desktop.

- `rapid-react.blob` This model has been trained on the Rapid-React balls from the 2022 FIRST Competition. The blob file format is designed to run specifically on an *OpenVINO* device.

- `rapid-react-config.json` This is the configuration file needed to load the rapid-react model.  It includes the class labels and confidence level. 

### Running the inference script

To run the inference using an attached Raspberry Pi camera.  

    cd ${HOME}tensorrt_demos
    python3 trt_yolo_mjpeg.py --onboard 0 -m rapid-react

For a USB camera:    

    python3 trt_yolo_wpi.py --usb 1 -m rapid-react

You can display the output stream in a desktop gui window like this:  

    python3 trt_yolo_wpi.py --usb 1 -m rapid-react --gui
