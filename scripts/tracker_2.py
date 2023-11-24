#!/usr/bin/env python3
from geometry_msgs.msg import TransformStamped
import numpy as np
import rospy
from std_msgs.msg import Bool, Float32MultiArray, String
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import tf2_ros
import tf
from multi_object_tracker import Tracker
from ahold_product_detection.msg import ProductPoseArray
from ahold_product_detection.srv import ChangeProduct, ChangeProductResponse
from albert_database.srv import *
import time

class PoseData():

    def __init__(self) -> None:
        self.subscriber = rospy.Subscriber("/pose_estimation_results", ProductPoseArray, self.callback)
        self.previous_stamp = rospy.Time.now()

    def callback(self, data):
        self.data = data


class ProductTracker():
    def __init__(self) -> None:
        self.frequency = 30
        self.rate = rospy.Rate(self.frequency)

        self.pose_estimation = PoseData()
        self.tracker = Tracker(
            dist_threshold=0.10,
            max_frame_skipped=60,
            frequency=self.frequency,
            robot=True)

        self.change_product = rospy.Service("change_product", ChangeProduct, self.change_product_cb)
        self.publish_is_tracked = rospy.Publisher("~is_tracked", Bool, queue_size=10)
        self.is_tracked = Bool(False)
        self.detection_class_pub = rospy.Publisher("/detection_class", String, queue_size=1)
        self.database_client = rospy.ServiceProxy('get_product_info', productInfo)
        # self.update_detector = rospy.ServiceProxy('/detection_class', String)

        self.track = False
        self.tracked_product = ""

    def change_product_cb(self, request):
        rospy.loginfo(f"Received request to change tracked product from {self.tracked_product} to {request.product_name}")

        if request.product_name == "":
            rospy.loginfo("Disabling tracking")
            self.track = False
            self.tracked_product = request.product_name
        elif self.tracked_product == request.product_name:
            rospy.loginfo("Product is already tracked")
            self.track = True
        else:
            try:
                rospy.loginfo(f"Getting info from database for {request.product_name}")
                self.store_name = rospy.get_param("/store_name")
                response = self.database_client(request.product_name, self.store_name)

                rospy.loginfo(f"Updating detector to detect {request.product_name}")
                self.detection_class_pub.publish(String(request.product_id))

                rospy.loginfo(f"Resetting tracker to track {request.product_name}")
                self.tracker.reset()
                self.tracker.shelf_angle = response.shelf_ort * (np.pi/180)
                self.track = True
            except Exception as e:
                rospy.logerr(f"Failed to change tracked product, see {e}")
                return ChangeProductResponse(success=False)

        return ChangeProductResponse(success=True)

    def run(self):
        while not rospy.is_shutdown():
            try:
                stamp = self.pose_estimation.data.header.stamp
                product_poses = self.pose_estimation.data
                xyz_detections = [[p.x, p.y, p.z, p.theta, p.phi, p.psi] for p in product_poses.poses]

                # labels are always the same, since we only keep track of the requested product
                labels = [-1 for _ in product_poses.poses]

                scores = [p.score for p in product_poses.poses]
                
                if self.pose_estimation.previous_stamp.to_sec() == stamp.to_sec():
                    raise ValueError("New data has not been received... track with no measurements")
                self.pose_estimation.previous_stamp = stamp

            except Exception as e:
                rospy.logerr(f"not executing tracking loop because of: {e}")
                xyz_detections = []
                labels = []
                scores = []

            # Track the detected products with Kalman Filter
            self.tracker.process_detections(xyz_detections, labels, scores)

            # Publish if tracked
            self.is_tracked.data = self.tracker.requested_product_tracked
            self.publish_is_tracked.publish(self.is_tracked)

            self.rate.sleep()

        return True

if __name__ == "__main__":
    rospy.init_node("product_tracker")
    product_tracker = ProductTracker()
    product_tracker.run()