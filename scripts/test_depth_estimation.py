#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from ahold_product_detection.srv import *
from pynput import keyboard
from copy import deepcopy
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
import pandas as pd
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
import os
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
import cv2
from scipy.optimize import leastsq
import multiprocessing
import os
import time

# test for flat and round object
# orientation and depth estimation tests
# 100 frames per distance and orientation
# test orientation 0, 15, 30 45 degrees left (UP??)
# test depth 30, 100 cm

# distance impact can be an argument for visual servoing (closer = better, thus VS better)
# camera impact angle degrees???
# network only detects front of the product
# do test where not front only as well? For presentation...
# draw bounding box so entire front fits
class CameraData:
    def __init__(self) -> None:
        # Setup ros subscribers and service
        self.depth_subscriber = rospy.Subscriber(
            "/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback
        )
        self.rgb_subscriber = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.rgb_callback
        )
        self.pointcould_subscriber = rospy.Subscriber(
            "/camera/depth/color/points", PointCloud2, self.pointcloud_callback
        )
        self.intrinsics_subscriber = rospy.Subscriber(
            "/camera/color/camera_info", CameraInfo, self.intrinsics_callback
        )

        self.pointcloud_msg = PointCloud2()
        rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image, timeout=10)
        rospy.wait_for_message("/camera/color/image_raw", Image, timeout=10)
        # rospy.wait_for_message("/camera/depth/color/points", PointCloud2, timeout=15)

    def depth_callback(self, data):
        self.depth_msg = data

    def pointcloud_callback(self, data):
        self.pointcloud_msg = data

    def rgb_callback(self, data):
        self.rgb_msg = data

    def intrinsics_callback(self, data):
        self.intrinsics = np.array(data.K).reshape((3, 3))

    @property
    def data(self):
        # TODO: timesync or check if the time_stamps are not too far apart (acceptable error)
        return (
            self.rgb_msg,
            self.depth_msg,
            self.pointcloud_msg,
            self.rgb_msg.header.stamp,
            self.intrinsics
        )



def read_message(cam):
    """Returns rgb image, depth image, camera intrinsics, and the corresponding frame id"""
    rgb_msg, depth_msg, pointcloud_msg, stamp, intrinsics = cam.data
    frame_id = pointcloud_msg.header.frame_id
    
    # Convert camera intrinsics to numpy array
    intrinsic_camera_matrix = intrinsics
    
    # Convert image messages to cv2 images
    bridge = CvBridge() 
    rgb_image = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='passthrough')
    depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return rgb_image, depth_image, intrinsic_camera_matrix, frame_id



class Keyboard_interpreter():
    def __init__(self):

        self.listener = keyboard.Listener(
                    on_press=self.on_press,
                    on_release=self.on_release)
        

        self.command = []

        self.listener.start()

    def on_press(self, key):
        try:
            if key.char == 'z':
                self.command.append('z')
            if key.char == 'c':
                self.command.append('c')    
            if key.char == 'e':
                self.command.append('e')

        except AttributeError:
            if key == keyboard.Key.esc:
                self.command.append('esc')
            if key == keyboard.Key.left:
                self.command.append('left')
            if key == keyboard.Key.right:
                self.command.append('right')
            if key == keyboard.Key.enter:
                self.command.append('enter')
            

    def on_release(self, key):
        pass
            

    def join(self):
        self.listener.join()

    def stop(self):
        self.listener.stop()

    def get_command(self):
        command = self.command
        self.command = [] # empty command queue
        return command
    

class DepthOrientationTester():

    def __init__(self, num_frames) -> None:
        
        self.num_frames = num_frames
        self.keyboard_interpreter = Keyboard_interpreter()
        ################################################################################
        ################################################################################
        self.distance = 1 # distance test in meters
        self.theta = 0 # theta angle test in degrees UPDOWN
        self.phi = 0 # phi angle test in degrees LEFTRIGHT
        self.object_structure = 'round' # object structure, flat or round        
        self.bounding_box_multipliers = [0.5, 0.75, 1, 1.25, 1.5] 
        
        ################################################################################
        ################################################################################
        self.pc_methods = [1000, 500, 250, 125]
        self.cam = CameraData()
        


    @staticmethod
    def update_image_screen(image):
        cv2.imshow('Image screen', image)
        # resizee here as well
        cv2.waitKey(1)

    def resize_bounding_box(self, bounding_box, multiplier):
        
        width = bounding_box[2] - bounding_box[0]
        height = bounding_box[3] - bounding_box[1]

        center_x = (bounding_box[0] + bounding_box[2])/2
        center_y = (bounding_box[1] + bounding_box[3])/2

        new_bounding_box = np.array([int(center_x - multiplier*width/2), int(center_y - multiplier*height/2), int(center_x + multiplier*width/2), int(center_y + multiplier*height/2)])
        return new_bounding_box

    def run(self):
        
        while not rospy.is_shutdown():

            # draw bounding box
            bbox_done = False
            while not bbox_done:
                bbox_done, original_bounding_box = self.draw_bounding_box()
                cv2.destroyAllWindows()
            for pc_method in self.pc_methods:
                self.results_distances_mean = []
                self.results_distances_median = []
                self.results_thetas = []
                self.results_phis = []
                self.pc_method = pc_method
                for bounding_box_multiplier in self.bounding_box_multipliers:
                    bounding_box = self.resize_bounding_box(original_bounding_box, bounding_box_multiplier)
                    self.median_distances = []
                    self.mean_distances = []
                    self.thetas = []
                    self.phis = []
                    for i in range(self.num_frames):
                        
                        rgb_image, depth_image, intrinsic_camera_matrix, frame_id = read_message(self.cam)
                        depth_bboxes = self.translate_rgb_bounding_boxes_to_depth(rgb_image, depth_image, [bounding_box])
                        depth_cut_out = depth_image[depth_bboxes[0][1]:depth_bboxes[0][3], depth_bboxes[0][0]:depth_bboxes[0][2]]
                        median_distance, mean_distance = self.get_median_distance_depth_data(depth_cut_out)
                        #print(median_distance)
                        self.median_distances.append(median_distance)
                        self.mean_distances.append(mean_distance)
                        orientation_estimator = OrientationEstimator(depth_image=depth_image, camera_intrinsics=intrinsic_camera_matrix, num_points=pc_method)

                        orientation = orientation_estimator.get_object_orientation(bounding_box)
                        #print('theta = ' + str(180*pose[3]/np.pi)) # up/down
                        #print('phi = ' + str(180*pose[4]/np.pi)) # left/right
                        self.thetas.append(180*orientation[0]/np.pi)
                        self.phis.append(180*orientation[1]/np.pi)
                        cv2.imshow("collecting data", rgb_image)
                        cv2.waitKey(1)
                    self.results_distances_median.append(self.median_distances)
                    self.results_distances_mean.append(self.mean_distances)
                    self.results_thetas.append(self.thetas)
                    self.results_phis.append(self.phis)
                    # process results
                    cv2.destroyAllWindows()
                    #print(self.calculate_mean_and_std(self.median_distances))
                
                #self.plot_distribution_distance(self.results_distances)
                #self.plot_distribution_theta(self.results_thetas) 
                #self.plot_distribution_phi(self.results_phis) 
                self.save_results(self.results_distances_mean, self.results_distances_median, self.results_phis, self.results_thetas)
            break



    def save_results(self, distances_mean, distances_median, phis, thetas):
        path = '/home/stijn/results_pose_estimation'
        with open(os.path.join(path, 'results_' + str(self.distance) + '_' + str(self.object_structure) + '_' + str(self.theta) + '_' + str(self.phi) + '_PC' +str(self.pc_method) + '.txt'), 'w') as f:
            
            f.write('mean distances;\n')
            for d in distances_mean:
                f.write("mean and standard deviation = " + str(self.calculate_mean_and_std(d)) + '\n')
                #f.write(str(d) + '\n')
            f.write('\n')

            f.write('median distances;\n')
            for d in distances_median:
                f.write("mean and standard deviation = " + str(self.calculate_mean_and_std(d)) + '\n')
                #f.write(str(d) + '\n')
            f.write('\n')
            
            f.write('thetas;\n')
            for d in thetas:
                f.write("mean and standard deviation = " + str(self.calculate_mean_and_std(d)) + '\n')
                #f.write(str(d) + '\n')
            f.write('\n')
            
            f.write('phis;\n')
            for d in phis:
                f.write("mean and standard deviation = " + str(self.calculate_mean_and_std(d)) + '\n')
                #f.write(str(d) + '\n')
            f.write('\n')
            


    def calculate_mean_and_std(self, data):
        mean = np.mean(data)
        std = np.std(data)
        return mean, std
    


    def plot_distribution_distance(self, results):

        binwidth = 0.001
        plt.xlabel("Estimated depth")
        plt.ylabel("Occurances")
        for i, r in enumerate(results):
            if i == 0:
                maximum = max(r)
                minimum = min(r)
            else:
                if maximum < max(r):
                    maximum = max(r)
                if minimum > min(r):
                    minimum = min(r)

        plot = sns.histplot(results, kde=False, alpha=0.5, bins=np.arange(minimum, maximum + binwidth, binwidth), multiple='dodge')
        plt.xticks(np.arange(minimum, maximum + binwidth, binwidth))
        plt.legend(title=self.bbox_mode ,labels=[str(np.abs(int(100*(1-i))))+'%' for i in self.bounding_box_multipliers])
        plt.show()
    
        


    def plot_distribution_theta(self, results):
        binwidth = 1
        plt.xlabel("Estimated theta")
        plt.ylabel("Occurances")
        for i, r in enumerate(results):
            if i == 0:
                maximum = max(r)
                minimum = min(r)
            else:
                if maximum < max(r):
                    maximum = max(r)
                if minimum > min(r):
                    minimum = min(r)

        plot = sns.histplot(results, kde=False, alpha=0.5, bins=np.arange(minimum, maximum + binwidth, binwidth), multiple='dodge')
        #line = plt.axvline(self.theta, color='k', linestyle='dashed', linewidth=2, label='True theta')
        plt.xticks(np.arange(minimum, maximum + binwidth, binwidth, dtype=np.int64))

        plt.legend(title=self.bbox_mode ,labels=[str(np.abs(int(100*(1-i))))+'%' for i in self.bounding_box_multipliers])
        plt.show()
        
    
    
    def plot_distribution_phi(self, results):
        binwidth = 1
        plt.xlabel("Estimated phi")
        plt.ylabel("Occurances")
        for i, r in enumerate(results):
            if i == 0:
                maximum = max(r)
                minimum = min(r)
            else:
                if maximum < max(r):
                    maximum = max(r)
                if minimum > min(r):
                    minimum = min(r)

        plot = sns.histplot(results, kde=False, alpha=0.5, bins=np.arange(minimum, maximum + binwidth, binwidth), multiple='dodge')
        #line = plt.axvline(self.phi, color='k', linestyle='dashed', linewidth=2, label='True phi')
        plt.xticks(np.arange(minimum, maximum + binwidth, binwidth, dtype=np.int64))

        plt.legend(title=self.bbox_mode ,labels=[str(np.abs(int(100*(1-i))))+'%' for i in self.bounding_box_multipliers])
        plt.show()
        
        
    def convert_PointCloud2_from_numpy_array(self, array, frame_id):
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

    def get_orientation_plane(self, fit):
        """Returns euler angles of a fitted plane"""
        theta = np.arctan(fit[1]) # rotation around camera x axis
        phi = np.arctan(fit[0])  # rotation around camera y axis
    
        return [float(theta), float(-phi), float(0)] # [x, y, z] rotations

    def fit_plane(self, points):
        """Returns function values of plane fitted to a set of points (pointcloud)"""
        # Fit plane to function: ax + by + c = z (so goal: get a, b and c)
        A = np.hstack((points[:,:2], np.ones((len(points), 1)))) # xy1 vectors (= [x, y, 1])
        b = points[:, 2] # z values

        fit = np.linalg.pinv(A) @ b # [a, b, c]

        # Calculate error
        errors = b - A @ fit
        residual = np.linalg.norm(errors)

        return fit
    

    def estimate_pointcloud_orientation_with_plane(self, pointcloud):
        """Returns the orientation of pointcloud by fitting a plane to it"""
        fit = self.fit_plane(pointcloud)
        orientation = self.get_orientation_plane(fit)
        return orientation

    def estimate_pose_object(self, depth_image, depth_bounding_box, intrinsic_camera_matrix, frame_id):
        """Returns pose message and pointcloud message"""
        # Calculate middle of bounding box (which is the center of the object's surface)
        xyz_vector = self.get_grasp_coordinates(depth_image, depth_bounding_box, intrinsic_camera_matrix)
        
        # Estimate the pointcloud that corresponds to the object's surface only
        pointcloud, pointcloud_message = self.depth_bounding_box_to_pointcloud(depth_image, depth_bounding_box, intrinsic_camera_matrix, frame_id)
        
        # Estimate orientation of object using a plane fit around the object's surface pointcloud
        orientation = self.estimate_pointcloud_orientation_with_plane(pointcloud)
        
        pose = xyz_vector.tolist() + orientation

        

        return pose
    
    def get_grasp_coordinates(self, depth_image, depth_bounding_box, intrinsic_camera_matrix):
        """Returns xyz vector of the center of the bounding box (which is at the surface of the object)"""

        # Get depth data bounding box
        depth_data_bounding_box = depth_image[depth_bounding_box[1]:depth_bounding_box[3], depth_bounding_box[0]:depth_bounding_box[2]]

        # Get median of bounding box as estimated z coordinate
        median_z, mean_z = self.get_median_distance_depth_data(depth_data_bounding_box)

        # Get bounding box center pixels
        bbox_center_u = int((depth_bounding_box[2] + depth_bounding_box[0])/2)
        bbox_center_v = int((depth_bounding_box[3] + depth_bounding_box[1])/2)

        # Calculate xyz vector with pixels (u, v) and camera intrinsics
        xyz_vector = self.translate_pixel_coordinates_to_cartesian_coordinates_in_camera_frame(bbox_center_u, bbox_center_v, median_z, intrinsic_camera_matrix)
        return xyz_vector
    

    def translate_pixel_coordinates_to_cartesian_coordinates_in_camera_frame(self, u, v, estimated_z, intrinsic_camera_matrix):
        """Returns estimated xyz vector of each bounding box, representing the center of the bounding box at the object surface"""
        
        pixel_vector = np.array([u, v, 1])
        scaled_xyz_vector = np.linalg.inv(intrinsic_camera_matrix) @ pixel_vector.T 
        xyz_vector = estimated_z * scaled_xyz_vector

        return xyz_vector


    def depth_bounding_box_to_pointcloud(self, depth_image, depth_bounding_box, intrinsic_camera_matrix, frame_id):
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
        pointcloud_message = self.convert_PointCloud2_from_numpy_array(np.hstack((filtered_points, filtered_z_values.reshape(len(filtered_z_values), 1))), frame_id)

        return pointcloud, pointcloud_message


    
    def translate_rgb_bounding_boxes_to_depth(self, rgb_image, depth_image, xyxy_bounding_boxes):
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



    def get_median_distance_depth_data(self, depth_data):
        """Returns median of depth data in meters"""
        median_z = np.median(depth_data)/1000 #
        mean_z = np.mean(depth_data)/1000
        return median_z, mean_z
    


    def draw_bounding_box(self):
        # Request new message from rgbd_processor
        
        rgb_image, depth_image, intrinsic_camera_matrix, frame_id = read_message(self.cam)
        self.image = rgb_image
        self.update_image_screen(rgb_image)
        editing = True
        self.new_bounding_boxes = []
        editing = self.draw_new_bounding_box(editing)
        if editing == False:
            done = True
            bounding_box = self.new_bounding_boxes[0]
        if len(self.new_bounding_boxes) > 1:
            raise ValueError("Too many boxes drawn, use 1 only!")
        else:
            return done, bounding_box

    def contain_bounding_box_within_frame(self, x, y, image):
        # make sure bounding box stays within bounds
        # image shape openCV: (height, width, channels)
        if x < 0:
            x = 0
        elif x > image.shape[1]:
            x = image.shape[1]
        if y < 0:
            y = 0
        elif y > image.shape[0]:
            y = image.shape[0]
        
        return x, y
    
    def draw_new_bounding_boxes(self, new_bounding_boxes, image):

        for bounding_box in new_bounding_boxes:
            cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0,255,255), 1)
        return image
    def draw_current_bounding_box(self, end_bbox_x, end_bbox_y):
        
        cv2.rectangle(self.drawing_image, (self.initial_bbox_x, self.initial_bbox_y), (end_bbox_x, end_bbox_y), (0,0,0), 1)

    def draw_helpers(self, mouse_x, mouse_y, image):
        # lines
        image_with_lines = deepcopy(self.image)
        image_with_lines = cv2.line(image_with_lines, (mouse_x, 0), (mouse_x, image_with_lines.shape[0]), (255,0,0), 1)
        image_with_lines = cv2.line(image_with_lines, (0, mouse_y), (image_with_lines.shape[1], mouse_y), (255,0,0), 1)

        # bounding boxes
        image_with_lines = self.draw_new_bounding_boxes(self.new_bounding_boxes, image_with_lines)
        return image_with_lines


    def draw_new_bounding_box(self, editing):
        # Connect the mouse button to our callback function
        

        self.drawing = False
        self.initial_bbox_x, self.initial_bbox_y = -1, -1
        self.draw_with_clear_view = False
        self.drawing_image = deepcopy(self.image)
        cv2.setMouseCallback("Image screen", self.annotate_bounding_box)

        # display the window
        while editing:            
            self.update_image_screen(self.drawing_image)
            
            command = self.keyboard_interpreter.get_command()

            if len(command) > 0 and command[0] == 'esc':
                raise ValueError("Bounding box drawn incorrect, start over")
            if len(command) > 0 and command[0] == 'enter':
                editing = False
        return editing
    
    
    def annotate_bounding_box(self, event, mouse_x, mouse_y, flags, param):
 
        # start drawing bounding box
        if event == cv2.EVENT_MBUTTONDOWN:
            self.drawing = True
            self.initial_bbox_x = mouse_x
            self.initial_bbox_y = mouse_y
            end_bbox_x = mouse_x
            end_bbox_y = mouse_y
        
        # check mouse while drawing bounding box
        if event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                
                end_bbox_x = mouse_x
                end_bbox_y = mouse_y

                # contain bounding box within frame
                end_bbox_x, end_bbox_y = self.contain_bounding_box_within_frame(end_bbox_x, end_bbox_y, self.drawing_image)

        # stop drawing bounding box
        if event == cv2.EVENT_MBUTTONUP:
            
            
            end_bbox_x = mouse_x
            end_bbox_y = mouse_y

            
            # contain bounding box within frame
            end_bbox_x, end_bbox_y = self.contain_bounding_box_within_frame(end_bbox_x, end_bbox_y, self.drawing_image)

            # save annotation
            
            if end_bbox_x > self.initial_bbox_x and end_bbox_y > self.initial_bbox_y:
                new_bounding_box = np.array([self.initial_bbox_x, self.initial_bbox_y, end_bbox_x, end_bbox_y])
            elif end_bbox_x < self.initial_bbox_x and end_bbox_y > self.initial_bbox_y:
                new_bounding_box = np.array([end_bbox_x, self.initial_bbox_y, self.initial_bbox_x, end_bbox_y])
            elif end_bbox_x > self.initial_bbox_x and end_bbox_y < self.initial_bbox_y:
                new_bounding_box = np.array([self.initial_bbox_x, end_bbox_y, end_bbox_x, self.initial_bbox_y])
            else:
                new_bounding_box = np.array([end_bbox_x, end_bbox_y, self.initial_bbox_x, self.initial_bbox_y])
            self.new_bounding_boxes.append(new_bounding_box)
            self.drawing = False

        if True:
            self.drawing_image = self.draw_helpers(mouse_x, mouse_y, self.drawing_image)
            
        if self.drawing:
            self.draw_current_bounding_box(end_bbox_x, end_bbox_y)

def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                        if np.isnan(output[x, y]):
                            raise ValueError("NAN!!!!!!!!!!!")
                except:
                    break

    return output


class OrientationEstimator:

    def __init__(self, depth_image, camera_intrinsics, num_points) -> None:
        self.depth_image = depth_image
        self.camera_intrinsics = camera_intrinsics
        self.num_points = num_points


    def get_bbox_pointcloud(self, bounding_box):
        """Returns pointcloud and pointcloud message, created from the depth data of the bounding box"""

        # Get depth data of bounding box
        bbox_depth_data =  self.depth_image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]

        # Create grid of all pixels for efficient matrix calculation
        uu, vv = np.meshgrid(np.arange(bounding_box[0], bounding_box[2]), np.arange(bounding_box[1], bounding_box[3]))
        uv_vector = np.vstack((uu.flatten(), vv.flatten(), np.ones(len(vv.flatten()))))

        # Get all z values that correspond with the pixels, format them to meters
        z_values = bbox_depth_data.flatten()/1000

        # Calculate pointcloud
        scaled_points = np.linalg.inv(self.camera_intrinsics) @ uv_vector
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

        filtered_points = nonzero_points[np.where(band_pass_z_values!=0)]

        self.pointcloud = filtered_points


    def get_bbox_pointcloud_grid(self, bounding_box, step_size):
        """Returns pointcloud and pointcloud message, created from the depth data of the bounding box"""

        # Create grid of all pixels for efficient matrix calculation
        us = np.arange(bounding_box[0], bounding_box[2], step_size)
        vs = np.arange(bounding_box[1], bounding_box[3], step_size)
        #print(bounding_box)
        #print(us)
        uu, vv = np.meshgrid(us, vs)
        uv_vector = np.vstack((uu.flatten(), vv.flatten(), np.ones(len(vv.flatten()))))
        
        zs = []

        for u in us: 
            for v in vs:
                zs.append(self.depth_image[v,u])

        
        # Get all z values that correspond with the pixels, format them to meters
        z_values = np.array(zs)/1000

        # Calculate pointcloud
        scaled_points = np.linalg.inv(self.camera_intrinsics) @ uv_vector
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

        filtered_points = nonzero_points[np.where(band_pass_z_values!=0)]

        self.pointcloud = filtered_points

    
    def get_bbox_pointcloud_random(self, bounding_box, num_points):
        """Returns pointcloud and pointcloud message, created from the depth data of the bounding box"""
        #TODO: remove duplicate points
        # Get depth data of bounding box
        #bbox_depth_data =  self.depth_image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]

        zs = []
        uu = np.random.randint(bounding_box[0],bounding_box[2]-1, num_points)
        vv = np.random.randint(bounding_box[1],bounding_box[3]-1, num_points)
        uv_vector = np.vstack((uu, vv, np.ones(len(vv))))
        
        for i, u in enumerate(uu): 
            zs.append(self.depth_image[vv[i],u])
        
        # Get all z values that correspond with the pixels, format them to meters
        z_values = np.array(zs)/1000

        # Calculate pointcloud
        scaled_points = np.linalg.inv(self.camera_intrinsics) @ uv_vector
        points = z_values * scaled_points

        # Only consider points that have depth data != 0 (some depth data is inaccurate)
        nonzero_points = points.T[np.where(z_values!=0)]
        nonzero_z_values = z_values.T[np.where(z_values!=0)]

        # Only consider points that are close to the median of the pointcloud,
        # because the bounding box also includes some background points, which 
        # must be removed to include object points only
        bound = 0.025   # meters

        lower_bound = np.where(nonzero_z_values > np.median(nonzero_z_values) - bound, 1, 0)
        upper_bound = np.where(nonzero_z_values < np.median(nonzero_z_values) + bound, 1, 0)

        band_pass = lower_bound * upper_bound
        band_pass_z_values = band_pass * nonzero_z_values

        filtered_points = nonzero_points[np.where(band_pass_z_values!=0)]

        self.pointcloud = filtered_points

    
    def get_pointcloud_orientation_with_plane_fit(self):
        """Returns the orientation of pointcloud by fitting a plane to it"""
        fit = self.fit_plane(self.pointcloud)
        orientation = self.get_orientation_plane(fit)
        return orientation


    @staticmethod
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


    @staticmethod
    def fit_cylinder(points):
        """
        p is initial values of the parameter;
        p[0] = Xc, x coordinate of the cylinder centre
        p[1] = Yc, y coordinate of the cylinder centre
        p[2] = alpha, rotation angle (radian) about the x-axis
        p[3] = beta, rotation angle (radian) about the y-axis
        p[4] = r, radius of the cylinder
        """  
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]
        p = np.array([np.median(x),np.median(y),0,0,0.3])  # initial guess
        
        fitfunc = lambda p, x, y, z: (- np.cos(p[3])*(p[0] - x) - z*np.cos(p[2])*np.sin(p[3]) - np.sin(p[2])*np.sin(p[3])*(p[1] - y))**2 + (z*np.sin(p[2]) - np.cos(p[2])*(p[1] - y))**2 #fit function
        errfunc = lambda p, x, y, z: fitfunc(p, x, y, z) - p[4]**2 #error function 

        est_p , success = leastsq(errfunc, p, args=(x, y, z), maxfev=1)

        return est_p


    @staticmethod
    def get_orientation_plane(fit):
        """Returns euler angles of a fitted plane"""
        theta = np.arctan(fit[1]) # rotation around camera x axis
        phi = np.arctan(fit[0])  # rotation around camera y axis
        
        return np.array([float(theta), float(-phi), float(0)]) # [x, y, z] rotations

    def get_object_orientation(self, bounding_box):
        small_bounding_box = self.resize_bounding_box(bounding_box, 0.5) # half the size of the bounding box
        #self.get_bbox_pointcloud(small_bounding_box)
        self.get_bbox_pointcloud_random(bounding_box, num_points=self.num_points) # [1000, 500, 250, 125]
        #self.get_bbox_pointcloud_grid(small_bounding_box, step_size=2)
        orientation = self.get_pointcloud_orientation_with_plane_fit()
        return orientation.tolist()
    
    def resize_bounding_box(self, bounding_box, multiplier):
        width = bounding_box[2] - bounding_box[0]
        height = bounding_box[3] - bounding_box[1]

        center_x = (bounding_box[0] + bounding_box[2])/2
        center_y = (bounding_box[1] + bounding_box[3])/2

        new_bounding_box = np.array([int(center_x - multiplier*width/2), int(center_y - multiplier*height/2), int(center_x + multiplier*width/2), int(center_y + multiplier*height/2)])
        return new_bounding_box
    

def run(camera):
    try:
        rgb_msg, depth_msg, pointcloud_msg, time_stamp, camera_intrinsics = camera.data
    except Exception as e:
        rospy.logerr(f"Couldn't read camera data", e)
        return
    return rgb_msg, depth_msg, camera_intrinsics

if __name__ == "__main__":
    num_frames = 100
    rospy.init_node('DepthOrientationTester', anonymous=False)
    d = DepthOrientationTester(num_frames=num_frames)
    d.run()