# FRC-OAK-Deployment-Models
This repository stores *Deep Learning* models and scripts that can be deployed on the Raspberry Pi or Jetson Nano. This example uses an attached [Luxonis OAK](https://shop.luxonis.com/products/1098obcenclosure) camera.  The OAK camera has an imbedded TPU the runs on the [Intel OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html). The OAK camera's TPU runs the inference model, taking the processing off of the host processor.  

Before deploying these models the Raspberry Pi must have the WPILibPi Romi image installed.  If running on a Jetson Nano that must be installed with Jetpack 4.6.  The main files are included in this repository are as follows:

- `oak_yolo_spacial.py`  This script runs inference on a Yolo model and outputs detected objects with a label, bounding boxes and their X, Y, Z coordinates from the camera.  The script will display its output in a Web browser at `<server IP address:8080` and also places all of the data into the *WPILib* Network Tables. If you're running this within a desktop environment you can also use the `--gui` option to display the output in a gui window.

- `rapid-react.blob` This model has been trained on the Rapid-React balls from the 2022 FIRST Competition. The blob file format is designed to run specifically on an *OpenVINO* device.

- `rapid-react-config.json` This is the configuration file needed to load the rapid-react model.  It includes the class labels and confidence level. 

### Running the Inference Script

To run the inference script using an OAK Depth camera:

    cd ${HOME}/depthai-python/examples/FRC-OAK-Deployment-Models
    python oak_yolo_spacial.py -m rapid-react

The streamed camera output can be viewed from `<Your server IP address>:8080`.  
> Note: The camera stream does not work in a Safari browser, use Chrome or Firefox.

To run the inference script within a desktop GUI window:

    python oak_yolo_spacial.py -m rapid-react --gui
