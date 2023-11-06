#!/usr/bin/env python3
import rospy
from ahold_product_detection.srv import *
from cv_bridge import CvBridge
import ultralytics
from ultralytics.yolo.utils.plotting import Annotator
import cv2
import os
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header, Int32
from ahold_product_detection.msg import FloatList, PointCloudList, ProductClass, ProductList
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import tf2_ros
import tf
from geometry_msgs.msg import TransformStamped, PoseStamped
from multiObjectTracker import Tracker


def requestRGBD_client():
    """Returns message received from rgbd_processor"""

    # Get RGBD data when ready from the rgbd_processor node
    rospy.wait_for_service('request_rgbd')
    
    try:
        request_rgbd = rospy.ServiceProxy('request_rgbd', RequestRGBD)
        response = request_rgbd(0)
        
        if response.data_available:
            return response
        else:
            return None
    
    except rospy.ServiceException as e:
        rospy.logwarn("Service call failed: %s"%e)



def predict(model, rgb_image):
    """Returns predicted bounding boxes, possible classes, and predicted scores for each bounding box"""
    
    # Perform object detection on an image using the model (YOLO Neural Network)
    results = model.predict(source=rgb_image, show=False, save=False, verbose=False)

    # Get resulting bounding boxes defined as (x, y, width, height)
    bounding_boxes = results[0].boxes.xyxy.cpu().numpy()
    
    # Get confidence values (scores) of each bounding box
    scores = results[0].boxes.conf.cpu().numpy()

    # Get names of all classes that can be detected
    names = results[0].boxes.cls.cpu().numpy()

    return results, bounding_boxes, names, scores



def translate_rgb_bounding_boxes_to_depth(rgb_image, depth_image, xyxy_bounding_boxes):
    """returns bounding boxes translated to the depth image"""

    # Depth is aligned with rgb, but sizes differ    
    # cv2 image shapes are defined as (height, width, channels)
    width_scale = depth_image.shape[1] / rgb_image.shape[1] 
    height_scale = depth_image.shape[0] / rgb_image.shape[0]
    
    depth_bounding_boxes = []
    
    for bounding_box in xyxy_bounding_boxes:

        # Resize bounding box to depth image
        x_start = int(width_scale * bounding_box[0])
        x_end = int(width_scale * bounding_box[2])
        y_start = int(height_scale * bounding_box[1])
        y_end = int(height_scale * bounding_box[3])

        depth_bounding_box = np.array([x_start, y_start, x_end, y_end])

        depth_bounding_boxes.append(depth_bounding_box)
    
    return depth_bounding_boxes



def get_median_distance_depth_data(depth_data):
    """Returns median of depth data in meters"""
    mean_z = np.median(depth_data)/1000 #
    return mean_z



def get_grasp_coordinates(depth_image, depth_bounding_box, intrinsic_camera_matrix):
    """Returns xyz vector of the center of the bounding box (which is at the surface of the object)"""

    # Get depth data bounding box
    depth_data_bounding_box = depth_image[depth_bounding_box[1]:depth_bounding_box[3], depth_bounding_box[0]:depth_bounding_box[2]]

    # Get median of bounding box as estimated z coordinate
    median_z = get_median_distance_depth_data(depth_data_bounding_box)

    # Get bounding box center pixels
    bbox_center_u = int((depth_bounding_box[2] + depth_bounding_box[0])/2)
    bbox_center_v = int((depth_bounding_box[3] + depth_bounding_box[1])/2)

    # Calculate xyz vector with pixels (u, v) and camera intrinsics
    xyz_vector = translate_pixel_coordinates_to_cartesian_coordinates_in_camera_frame(bbox_center_u, bbox_center_v, median_z, intrinsic_camera_matrix)
    return xyz_vector



def translate_pixel_coordinates_to_cartesian_coordinates_in_camera_frame(u, v, estimated_z, intrinsic_camera_matrix):
    """Returns estimated xyz vector of each bounding box, representing the center of the bounding box at the object surface"""
    
    pixel_vector = np.array([u, v, 1])
    scaled_xyz_vector = np.linalg.inv(intrinsic_camera_matrix) @ pixel_vector.T 
    xyz_vector = estimated_z * scaled_xyz_vector

    return xyz_vector



def depth_bounding_box_to_pointcloud(depth_image, depth_bounding_box, intrinsic_camera_matrix, frame_id):
    """Returns pointcloud and pointcloud message, created from the depth data of the bounding box"""

    # Get depth data of bounding box
    product_depth_data =  depth_image[depth_bounding_box[1]:depth_bounding_box[3], depth_bounding_box[0]:depth_bounding_box[2]]

    # Create grid of all pixels for efficient matrix calculation
    uu, vv = np.meshgrid(np.arange(depth_bounding_box[0], depth_bounding_box[2]), np.arange(depth_bounding_box[1], depth_bounding_box[3]))
    uv_vector = np.vstack((uu.flatten(), vv.flatten(), np.ones(len(vv.flatten()))))

    # Get all z values that correspond with the pixels, format them to meters
    z_values = product_depth_data.flatten()/1000

    # Calculate pointcloud
    scaled_points = np.linalg.inv(intrinsic_camera_matrix) @ uv_vector
    points = z_values * scaled_points
    
    # Only consider points that have depth data != 0 (some depth data is inaccurate)
    nonzero_points = points.T[np.where(z_values!=0)]
    nonzero_z_values = z_values.T[np.where(z_values!=0)]

    # Only consider points that are close to the median of the pointcloud,
    # because the bounding box also includes some background points, which 
    # must be removed to include object points only
    bound = 0.025     # meters

    lower_bound = np.where(nonzero_z_values > np.median(nonzero_z_values) - bound, 1, 0)
    upper_bound = np.where(nonzero_z_values < np.median(nonzero_z_values) + bound, 1, 0)

    band_pass = lower_bound * upper_bound
    band_pass_z_values = band_pass * nonzero_z_values

    filtered_z_values = band_pass_z_values.T[np.where(band_pass_z_values!=0)]
    filtered_points = nonzero_points[np.where(band_pass_z_values!=0)]

    pointcloud = filtered_points
    
    # Generate pointcloud ROS message for visualization
    pointcloud_message = convert_PointCloud2_from_numpy_array(np.hstack((filtered_points, filtered_z_values.reshape(len(filtered_z_values), 1))), frame_id)

    return pointcloud, pointcloud_message



def estimate_pointcloud_orientation_with_plane(pointcloud):
    """Returns the orientation of pointcloud by fitting a plane to it"""
    fit = fit_plane(pointcloud)
    orientation = get_orientation_plane(fit)
    return orientation



def fit_plane(points):
    """Returns function values of plane fitted to a set of points (pointcloud)"""
    # Fit plane to function: ax + by + c = z (so goal: get a, b and c)
    A = np.hstack((points[:,:2], np.ones((len(points), 1)))) # xy1 vectors (= [x, y, 1])
    b = points[:, 2] # z values

    fit = np.linalg.pinv(A) @ b # [a, b, c]

    # Calculate error
    errors = b - A @ fit
    residual = np.linalg.norm(errors)

    return fit



def get_orientation_plane(fit):
    """Returns euler angles of a fitted plane"""
    theta = np.arctan(fit[1]) # rotation around camera x axis
    phi = np.arctan(fit[0])  # rotation around camera y axis
    
    return [float(theta), float(-phi), float(0)] # [x, y, z] rotations



def convert_PointCloud2_from_numpy_array(array, frame_id):
    """Returns PointCloud2 object made from a numpy array"""
    array = array.astype(np.float32)
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              PointField('intensity', 12, PointField.FLOAT32, 1)]

    header = Header()
    header.frame_id = frame_id
    header.stamp = rospy.Time.now()

    pc2 = point_cloud2.create_cloud(header, fields, array)
    return pc2



def read_message(message):
    """Returns rgb image, depth image, camera intrinsics, and the corresponding frame id"""
    image_msg, pointcloud_msg, camera_info_msg = message.images, message.pointcloud, message.intrinsics
    frame_id = pointcloud_msg.header.frame_id
    
    # Convert camera intrinsics to numpy array
    intrinsic_camera_matrix = np.array(camera_info_msg.K).reshape((3,3))
    
    # Convert image messages to cv2 images
    bridge = CvBridge() 
    rgb_image = bridge.imgmsg_to_cv2(image_msg[0], desired_encoding='passthrough')
    depth_image = bridge.imgmsg_to_cv2(image_msg[1], desired_encoding='passthrough')

    return rgb_image, depth_image, intrinsic_camera_matrix, frame_id



def convert_pose(pose, frame, listener):
    x, y, z, theta, phi, psi = pose
    
    # make tf PoseStamped for transformation
    p = PoseStamped()
    p.header.stamp = rospy.Time(0)
    p.header.frame_id = "camera_color_optical_frame"
    p.pose.position.x = x
    p.pose.position.y = y
    p.pose.position.z = z
    q = quaternion_from_euler(theta, phi, 0)
    p.pose.orientation.x = q[0]
    p.pose.orientation.y = q[1]
    p.pose.orientation.z = q[2]
    p.pose.orientation.w = q[3]

    # Transform pose to desired frame
    p_t = listener.transformPose(frame, p)
    
    # Convert back to original format
    q_t = [p_t.pose.orientation.x,
           p_t.pose.orientation.y,
           p_t.pose.orientation.z,
           p_t.pose.orientation.w]
    
    euler = euler_from_quaternion(q_t)

    new_pose = [p_t.pose.position.x,
                p_t.pose.position.y, 
                p_t.pose.position.z, 
                euler[0],
                euler[1],
                euler[2]]
    
    return new_pose



def estimate_pose_object(depth_image, depth_bounding_box, intrinsic_camera_matrix, frame_id, name, score, listener, robot):
    """Returns pose message and pointcloud message"""
    # Calculate middle of bounding box (which is the center of the object's surface)
    xyz_vector = get_grasp_coordinates(depth_image, depth_bounding_box, intrinsic_camera_matrix)
    
    # Estimate the pointcloud that corresponds to the object's surface only
    pointcloud, pointcloud_message = depth_bounding_box_to_pointcloud(depth_image, depth_bounding_box, intrinsic_camera_matrix, frame_id)
    
    # Estimate orientation of object using a plane fit around the object's surface pointcloud
    orientation = estimate_pointcloud_orientation_with_plane(pointcloud)
    
    pose = xyz_vector.tolist() + orientation

    if robot:
        frame = 'base_link'
        
    else:
        frame = 'camera_color_optical_frame'
    pose = convert_pose(pose, frame, listener)
    rospy.logwarn(frame)
    vector = [name] + [score] + pose

    pose_message = FloatList()
    pose_message.data = vector

    return pose_message, pointcloud_message



def estimate_pose_detected_objects(rgb_image, depth_image, rgb_bounding_boxes, intrinsic_camera_matrix, frame_id, names, scores, listener, robot):
    """Returns the poses and pointclouds of all detected objects"""
    pointcloud_messages = []
    products = ProductList()
    object_poses = []
    
    # Relate rgb_bounding_boxes to depth image
    depth_bounding_boxes = translate_rgb_bounding_boxes_to_depth(rgb_image, depth_image, rgb_bounding_boxes)
    
    for i, depth_bounding_box in enumerate(depth_bounding_boxes):
        name = names[i]
        score = scores[i]
        pose_message, pointcloud_message = estimate_pose_object(depth_image, 
                                                                depth_bounding_box, 
                                                                intrinsic_camera_matrix, 
                                                                frame_id,
                                                                name,
                                                                score,
                                                                listener,
                                                                robot)
        
        object_poses.append(pose_message)
        pointcloud_messages.append(pointcloud_message)
    products.data = object_poses
    return products, pointcloud_messages



def search_products(message, model, listener, robot):
    # Convert message to usable data
    rgb_image, depth_image, intrinsic_camera_matrix, frame_id = read_message(message)

    # Predict product bounding boxes and classes
    results, rgb_bounding_boxes, names, scores = predict(model, rgb_image)
    
    products, pointcloud_messages = estimate_pose_detected_objects(rgb_image, 
                                                                    depth_image, 
                                                                    rgb_bounding_boxes, 
                                                                    intrinsic_camera_matrix, 
                                                                    frame_id,
                                                                    names, 
                                                                    scores,
                                                                    listener,
                                                                    robot)
    
    frame = plot_detection_results(rgb_image, results, model)
    return frame, products, pointcloud_messages 



def plot_detection_results(frame, results, model):
    for r in results:
        
        annotator = Annotator(frame)
        
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
            
    frame = annotator.result()  
    return frame

def main():
    # Load yolo model weights for detection
    weight_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'yolo_model', 'shelf_detector.pt')
    model = ultralytics.YOLO(weight_path)
    # Initialize ros node
    rospy.init_node('Product_detector', anonymous=False)
    listener = tf.TransformListener()
    robot = rospy.get_param('/robot')
    ethernet = rospy.get_param('/ethernet')
    # print(robot)
    # print(ethernet)
    rospy.sleep(1)

    # Initialize kalman filter for object tracking
    dist_threshold = 2 # if objects are standing still, choose 0.1 (=10 cm), if movement, choose 0.5
    ethernet = True
    if ethernet:
        max_frame_skipped = 10
    else:
        max_frame_skipped = 3
    max_trace_length = 3
    tracker = Tracker(dist_threshold=dist_threshold, max_frame_skipped=max_frame_skipped, max_trace_length=max_trace_length, frequency=1, robot=robot)

    while not rospy.is_shutdown():
        # Request new message from rgbd_processor
        message = requestRGBD_client()
        
        if message != None:
            # Detect, localize and transform the detected products
            frame, products, pointcloud_messages = search_products(message, model, listener, robot)                       
            resized_frame = cv2.resize(frame, (1067, 600))
            # Track the detected products with Kalman Filter
            tracker.process_detections(products)
            cv2.imshow("Result", np.hstack((resized_frame, tracker.frame)))
            cv2.waitKey(1)
            


if __name__ == '__main__':
    main()