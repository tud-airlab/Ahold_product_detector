cmake_minimum_required(VERSION 3.0.2)
project(ahold_product_detection)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
  message_generation
  realsense2_camera
  tf2
  tf_conversions
  tf2_ros
  jsk_recognition_msgs
  pcl_conversions
  pcl_ros
  message_filters
)


add_message_files(
  FILES
  RotatedBoundingBox.msg
  Detection.msg
  ProductPose.msg
  ProductPoseArray.msg
)

## Generate services in the 'srv' folder
add_service_files(
   FILES
   ChangeProduct.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs  # Or other packages containing msgs
  sensor_msgs
  realsense2_camera
  geometry_msgs
  jsk_recognition_msgs
  ) 


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ahold_product_detection
#  CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

