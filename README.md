# Ahold Product Detector
This repository implements a product detector for products in the Albert Heijn supermarket. It is able to detect products using YoloV8 and track them using a Kalman Filter. 

To run ROS with python, either a venv virtual environment or the conda or mamba package manager can be used. Specific instructions for these can be found below.
<details>
<summary>Venv Instructions</summary>

### Install this ROS Package for Ubuntu

Before starting, make sure to have ROS Noetic installed: http://wiki.ros.org/noetic/Installation

After installation, initialize a workspace in order to build and run the ROS package. Do this by running the following commands:

```console
mkdir YOUR_WORKSPACE_NAME
cd YOUR_WORKSPACE_NAME
mkdir src
cd src
```
We want to have the ROS package inside the 'src' directory, so now that we are in here we can clone the repository:
```console
git clone git@github.com:stijnla/Ahold_product_detector.git
```
Now return to your workspace directory, source the ROS environment if you haven't done so, and build the package by running the following lines:
```console
cd ..
source /opt/ros/noetic/setup.bash
catkin build
source /devel/setup.bash
```
Now that we have setup the ROS package, it is time to setup the python virtual environment for the python dependencies. For simplicity, make sure that the location of your virtual environment is NOT in your workspace, so that catkin does not try to build your virtual environment. After choosing a location of desire, run the following commands to create and activate your virtual environment:

```console
python -m venv PATH_TO_YOUR_VIRTUAL_ENV
source PATH_TO_YOUR_VIRTUAL_ENV/bin/activate
```

Now we install the dependencies for python in this virtual environment so it does not interfere with any other projects:

```console
pip install numpy
pip install opencv-python
pip install ultralytics
pip install roslibpy
pip install scipy
```

If you want to rebuild the ROS package, you should first deactivate the python virtual environment, than clean the build and re-build the package. Run the following commands in the workspace directory to achieve this:

```console
deactivate
catkin clean
catkin build
source devel/setup.bash
```

</details>

<details>
<summary>Conda/Mamba Instructions</summary>

### Install this ROS Package for Ubuntu

Install a conda build of ROS and the dependencies for this product detector package by using this command:

```console
mkdir YOUR_WORKSPACE_NAME/src --parents
cd YOUR_WORKSPACE_NAME/src
git clone git@github.com:stijnla/Ahold_product_detector.git
cd Ahold_product_detector
conda env create -f environment.yml 
```

After installation, initialize a workspace in order to build and run the ROS package. Do this by running the following commands:

```console
cd YOUR_WORKSPACE_NAME
conda activate ros_env
catkin init
```
We want to have our ROS package inside the 'src' directory, so now that we are in here we can clone the repository:
```console
cd src
git clone git@github.com:stijnla/Ahold_product_detector.git
```

If you want to rebuild the ROS package you should clean the build directory and re-build the package. Run the following commands in the workspace directory to achieve this:
```console
conda activate ros_env
catkin clean
catkin build
```

</details>

### Run the code

In order to run the code, make sure that the ROS package is built and sourced. Then run the following command with a realsense camera attached to your system:

```console
roslaunch ahold_product_detection detect.launch
```

For just debugging without a robot baseframe, use

```console
roslaunch ahold_product_detection detect_without_robot.launch
```


And run the following if only the detection nodes are necessary if you are using a robot for example:

```console
roslaunch ahold_product_detection robot.launch
```
