#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool, Float32MultiArray
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import tf2_ros
import tf
from multi_object_tracker import Tracker
from ahold_product_detection.msg import ProductPoseArray
from ahold_product_detection.srv import ChangeProduct, ChangeProductResponse
import time

VELOCITY = True

class PoseData():

    def __init__(self) -> None:
        self.subscriber = rospy.Subscriber("/pose_estimation_results", ProductPoseArray, self.callback)
        self.previous_stamp = rospy.Time.now()

    def callback(self, data):
        self.data = data


class ProductTracker():
    def __init__(self) -> None:
        self.frequency = 30
        self.pose_estimation = PoseData()
        self.velocity = VELOCITY
        if self.velocity:
            self.tracker = Tracker(
                dist_threshold=1,
                max_frame_skipped=60,
                frequency=self.frequency,
                robot=False,
            )
        else:
            self.tracker = Tracker(
                dist_threshold=0.1,
                max_frame_skipped=240,
                frequency=self.frequency,
                robot=False)
        self.rate = rospy.Rate(self.frequency) # track products at 30 Hz
        self.change_product = rospy.Service("change_product", ChangeProduct, self.change_product_cb)
        self.publish_is_tracked = rospy.Publisher("~is_tracked", Bool, queue_size=10)
        self.publish_image_coordinates = rospy.Publisher("~assigned_detection_image_coordinates", Float32MultiArray, queue_size=10)
        self.is_tracked = Bool(False)

        self.measure = False
        self.count = 0
        

    def change_product_cb(self, request):
        rospy.loginfo(f"Changing tracked product from {self.tracker.requested_yolo_id} to {request.product_id}")
        self.tracker.requested_yolo_id = request.product_id
        self.measure = True
        return ChangeProductResponse(success=True)
    
    def run(self):
        try:
            stamp = self.pose_estimation.data.header.stamp
            product_poses = self.pose_estimation.data
            xyz_detections = [[p.x, p.y, p.z, p.theta, p.phi, p.psi] for p in product_poses.poses]
            labels = [p.label for p in product_poses.poses]
            scores = [p.score for p in product_poses.poses]
            
            if self.pose_estimation.previous_stamp.to_sec() == stamp.to_sec():
                raise ValueError("New data has not been received... track with no measurements")
            self.pose_estimation.previous_stamp = stamp
        except Exception as e:
            xyz_detections = []
            labels = []
            scores = []

        # Track the detected products with Kalman Filter
        self.tracker.process_detections(xyz_detections, labels, scores)

        # Publish if tracked
        self.is_tracked.data = self.tracker.requested_product_tracked
        self.publish_is_tracked.publish(self.is_tracked)

        # Publish the u, w image coordinates of the assigned detection
        self.assigned_detection_image_coordinates = [
            product_poses[self.tracker.assigned_track.latest_measurement_idx].u,
            product_poses[self.tracker.assigned_track.latest_measurement_idx].v
        ]
        self.publish_image_coordinates.publish(self.assigned_detection_image_coordinates)
        
        return True

if __name__ == "__main__":
    rospy.init_node("product_tracker")
    product_tracker = ProductTracker()
    t0 = time.time()
    while not rospy.is_shutdown():
        b = product_tracker.run()
        product_tracker.rate.sleep()
        # print(f"product tracking rate: {1/(time.time() - t0)}")
        t0 = time.time()
        if not b:
            break