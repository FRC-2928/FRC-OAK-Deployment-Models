# FRC-OAK-Deployment-Models
This repository stores *Deep Learning* models and scripts that can be deployed on the Raspberry Pi or Jetson Nano. This example uses an attached [Luxonis OAK](https://shop.luxonis.com/products/1098obcenclosure) camera.  The OAK camera has an imbedded TPU the runs on the [Intel OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html). The OAK camera's TPU runs the inference model, taking the processing off of the host processor.  

Before deploying these models the Raspberry Pi must have the WPILibPi Romi image installed.  If running on a Jetson Nano that must be installed with Jetpack 4.6.  The main files are included in this repository are as follows:

### OAK Camera Deployment for Jetson Nano
This section will show you how to deploy a Yolo object detection model on a Jetson Nano using the OAK camera.  The deployment will require the installation of the Depthai software. 

To prepare for the install read the [Depthai Jetson Install](https://docs.luxonis.com/projects/api/en/latest/install/#jetson) on the Luxonis site.  This documentation shows you how to configure a 4GB swap space on your Jetson Nano and setup a python pip `virtualenv`.  If you already have your device setup with those options you can just follow the next steps.

Create a virtual environment.  The following command will create and put you into the virtual environment.  

    mkvirtualenv depthAI -p python3

Download and install the dependency package for the Depthai software:

    sudo wget -qO- https://docs.luxonis.com/install_dependencies.sh | bash

Clone the `depthai` github repository.  This example clones it in your $HOME directory:

    git clone https://github.com/luxonis/depthai-python.git
    cd ${HOME}/depthai-python

Edit your `.bashrc` with the following line:

    echo "export OPENBLAS_CORETYPE=ARMV8" >> ~/.bashrc

Install requirements for depthAI:

    cd ${HOME}/depthai-python/examples
    python install_requirements.py      

### Installing the FRC Detection Scripts        
In order to run the OAK-D camera and display the detected objects you need to deploy a custom python script that is specific to the type of detection model that you want to deploy.  In our case, we're going to use the `YoloV4-Tiny` model.  The script will use a default model that detects various common objects and display the output to a Web URL. It'll also place information about objects that have been detected into Network Tables so a it can be used by your WPILib java program.

To deploy the script follow these steps:

- Clone the FRC models and python scripts from GitHub:

        cd ${HOME}/depthai-python/examples
        git lfs clone https://github.com/FRC-2928/FRC-OAK-Deployment-Models.git

- Install the python requirements for the FRC scripts. The requirements are `pkgconfig`, `robotpy-cscore`, and `Pillow`. `robotpy-cscore` installs the *WPI Network Tables*. This can take 10-15 minutes to install.

        cd ${HOME}/depthai-python/examples/FRC-OAK-Deployment-Models
        python3 -m pip install -r requirements.txt        

There is also a model supplied with the *FRC-OAK-Deployment-Models* package that detects the Rapid-React balls from the 2022 competition.

### Running the Inference Script
Ensure that you're in the depthAI environment:

    workon depthAI
    
To run the inference script using an OAK Depth camera:

    cd ${HOME}/depthai-python/examples/FRC-OAK-Deployment-Models
    python3 oak_yolo_spacial.py -m rapid-react

The streamed camera output can be viewed from `<Your server IP address>:8080`.  
> Note: The camera stream does not work in a Safari browser, use Chrome or Firefox.

To run the inference script within a desktop GUI window:

    python oak_yolo_spacial.py -m rapid-react --gui

### Scripts    
- `oak_yolo_spacial.py`  This script runs inference on a Yolo model and outputs detected objects with a label, bounding boxes and their X, Y, Z coordinates from the camera.  The script will display its output in a Web browser at `<server IP address:8080` and also places all of the data into the *WPILib* Network Tables. If you're running this within a desktop environment you can also use the `--gui` option to display the output in a gui window.

- `rapid-react.blob` This model has been trained on the Rapid-React balls from the 2022 FIRST Competition. The blob file format is designed to run specifically on an *OpenVINO* device.

- `rapid-react-config.json` This is the configuration file needed to load the rapid-react model.  It includes the class labels and confidence level. 


