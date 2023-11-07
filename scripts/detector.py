#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge
import ultralytics
from ultralytics.yolo.utils.plotting import Annotator
import os
import numpy as np
from multi_object_tracker import Tracker

# message and service imports
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, Image, CompressedImage
from jsk_recognition_msgs.msg import (
    BoundingBox,
    BoundingBoxArrayWithCameraInfo,
    BoundingBoxArray,
)
from ahold_product_detection.srv import *
from ahold_product_detection.msg import Detection, RotatedBoundingBox

from tf.transformations import quaternion_from_euler, euler_from_quaternion
import tf2_ros
import tf
from opencv_helpers import RotatedRectCorners, RotatedRect
from copy import copy
from rotation_compensation import RotationCompensation

from std_msgs.msg import Bool

product_mapping  = {
0: 8710400280507,
1: 8718907400718,
2: 8718907457583,
3: 8718907400701,
4: 8718906697560,
5: 8718907056274,
6: 8710400035862,
7: 8710400687122,
8: 8718907056311,
9: 8710400145981,
10: 8718907306744,
11: 8718907306775,
12: 8718907306737,
13: 8718907306751,
14: 8718907056298,
15: 8710400011668,
16: 3083681149630,
17: 3083681025484,
18: 3083681068146,
19: 3083681126471,
20: 3083681068122,
21: 8712800147008,
22: 8712800147770,
23: 8714100795699,
24: 8714100795774,
25: 8720600612848,
26: 8720600609893,
27: 8720600606731,
28: 8717662264382,
29: 8717662264368,
30: 87343267,
31: 8710400514107,
32: 8718906872844,
33: 8718907039987,
34: 8710400416395,
35: 8718907039963,
36: 5414359921711,
37: 8718906948631,
38: 8718265082151,
39: 8718906536265,
40: 8718951065826,
41: 3574661734712,
42: 8006540896778,
43: 8720181388774,
44: 90453656,
45: 90453533,
46: 5410013114697,
47: 5410013136149,
48: 80042556,
49: 80042563,
50: 8005110170324,
51: 8001250310415,
52: 8004690052044,
53: 8718906124066,
54: 8718906124073,
55: 9999,
}

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

    @property
    def data(self):
        # TODO: timesync or check if the time_stamps are not too far apart (acceptable error)
        return (
            self.rgb_msg,
            self.depth_msg,
            self.pointcloud_msg,
            self.rgb_msg.header.stamp,
        )


class ProductDetector:
    def __init__(self) -> None:
        self.camera = CameraData()
        self.rotation_compensation = RotationCompensation()
        self.rotate = True
        self.rate = rospy.Rate(30)
        weight_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "yolo_model",
            "best_complete.pt",
        )
        self.model = ultralytics.YOLO(weight_path)
        self.pub = rospy.Publisher("/detection_results", Detection, queue_size=10)
        self.pub_img = rospy.Publisher("/detection_image", Image, queue_size=10)
        self.pub_img_compressed = rospy.Publisher("/detection_image_compressed", CompressedImage, queue_size=10)
        self.pub_img_barcode = rospy.Publisher("/detection_image_barcode", Image, queue_size=10)

        self.bridge = CvBridge()

        self._currently_recording = False
        self._currently_playback = False
        self.start_tablet_subscribers()
        self._barcode = None
        self._check_barcode_timer = rospy.Timer(rospy.Duration(0.1), self.check_barcode_update)

        self._assigned_detection_subscriber = rospy.Subscriber('/product_tracker/assigned_detection_image_coordinates', Float32MultiArray, self.assigned_detection_cb)
        self._assigned_detection_uv = [-1, -1]

    def assigned_detection_cb(self, msg):
        self._assigned_detection_uv = list(msg.data)

    def start_playback_callback(self, msg):
        if msg.data:
            self._currently_playback = True

    def stop_playback_callback(self, msg):
        if msg.data:
            self._currently_playback = False

    def start_record_callback(self, msg):
        if msg.data:
            self._currently_recording = True
            try:
                self._barcode = rospy.get_param("/barcode")            
                rospy.loginfo(f"Start recording with barcode: {self._barcode}")
                rospy.loginfo(f"Barcode type: {type(self._barcode)}")  # Add this line
            except Exception as e:
                rospy.logerr(e)



    def stop_record_callback(self, msg):
        if msg.data:
            self._currently_recording = False


    def plot_detection_results(self, frame, results):
        for r in results:
            annotator = Annotator(frame)

            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[
                    0
                ]  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                annotator.box_label(b, self.model.names[int(c)])

        frame = annotator.result()

        cv2.imshow("Result", frame)
        cv2.waitKey(1)

    def show_rotated_results(self, image, boxes, angle):
        for box in boxes:
            centers_dims = [(int(box[2 * j]), int(box[2 * j + 1])) for j in range(2)]
            RotatedRect(
                image,
                centers_dims[0],
                centers_dims[1][0],
                centers_dims[1][1],
                -angle,
                (0, 0, 255),
                2,
            )
        cv2.imshow("rotated results", image)

    def generate_detection_message(self, time_stamp, boxes, scores, labels):
        detection_msg = Detection()
        detection_msg.header.stamp = time_stamp

        bboxes_list = []
        for bbox, label, score in zip(boxes, labels, scores):
            bbox_msg = RotatedBoundingBox()

            bbox_msg.x = int(bbox[0])
            bbox_msg.y = int(bbox[1])
            bbox_msg.w = int(bbox[2])
            bbox_msg.h = int(bbox[3])
            bbox_msg.label = int(label)
            bbox_msg.score = score

            bboxes_list.append(bbox_msg)

        detection_msg.detections = bboxes_list

        return detection_msg

    def start_tablet_subscribers(self):
        self._start_record_sub = rospy.Subscriber("/tablet/start_record", Bool, self.start_record_callback)
        self._stop_record_sub = rospy.Subscriber("/tablet/stop_record", Bool, self.stop_record_callback)
        self._playback_subscriber = rospy.Subscriber("/tablet/start_playback", Bool, self.start_playback_callback)
        self._stop_playback_subscriber = rospy.Subscriber("/tablet/stop_playback", Bool, self.stop_playback_callback)

    def check_barcode_update(self, event):
        new_barcode = rospy.get_param("/barcode", None)
        if new_barcode != self._barcode:
            rospy.loginfo(f"Barcode parameter changed from {self._barcode} to {new_barcode}")
            self._barcode = new_barcode


    def run(self):
        try:
            rgb_msg, depth_msg, pointcloud_msg, time_stamp = self.camera.data
        except Exception as e:
            rospy.logerr(f"Couldn't read camera data", e)
            return

        # rotate input
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        rotated_rgb_image = self.rotation_compensation.rotate_image(
            rgb_image, time_stamp
        )

        # # predict
        # results = self.model.track(
        #     source=rotated_rgb_image,
        #     persist=True,
        #     show=False,
        #     save=False,
        #     verbose=False,
        #     device=0,
        #     agnostic_nms=True,
        # )

        results = self.model.predict(
            source=rotated_rgb_image,
            show=False,
            save=False,
            verbose=False,
            device=0,
            agnostic_nms=True,
        )


        # inverse rotate output
        boxes, angle = self.rotation_compensation.rotate_bounding_boxes(
            results[0].boxes.xywh.cpu().numpy(), rgb_image
        )
        scores = results[0].boxes.conf.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()
        detection_results_msg = self.generate_detection_message(
            time_stamp, boxes, scores, labels
        )
        detection_results_msg.rgb_image = rgb_msg
        detection_results_msg.depth_image = depth_msg
        self.pub.publish(detection_results_msg)

        frame = rotated_rgb_image
        for r in results:
            annotator = Annotator(frame)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[
                    0
                ]  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                annotator.box_label(b, self.model.names[int(c)])

        frame = annotator.result()
        compressed_image = self.bridge.cv2_to_compressed_imgmsg(frame)
        self.pub_img_compressed.publish(compressed_image)
        raw_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.pub_img.publish(raw_image)


        # Draw colored boxes based on the barcode and tabled state
        
        # Check if recording is active and a barcode is set
        #draw_colored_boxes = self._currently_recording and self._barcode is not None
        # draw_colored_boxes = True
        for r in results:
            annotator2 = Annotator(rotated_rgb_image)
            boxes = r.boxes
            labels = r.boxes.cls.cpu().numpy()
            highest_score_index = None

            if self._barcode == None:
                highest_score_index = -1
            else:
                # Ensure the barcode is an integer for comparison
                barcode = int(self._barcode) if self._barcode.isdigit() else None

                # Filter boxes that match the barcode
                matching_boxes_indices = [i for i, label in enumerate(labels) if product_mapping.get(int(label), None) == barcode]

                # Find the highest score among the matching boxes
                if matching_boxes_indices:
                    highest_score_index = matching_boxes_indices[np.argmax(scores[matching_boxes_indices])]

            # closest to previous tracked detection

            # if len(boxes) > 0 and self._assigned_detection_uv != [-1, -1]:
            #     closest_detection_index = np.argmin([np.linalg.norm(box.xywh.cpu().numpy()[0][:2] - np.array(self._assigned_detection_uv)) for box in boxes])
            # else:
            #     closest_detection_index = -1

            for i, box in enumerate(boxes):
                label_index = int(labels[i])  # Convert label to int
                # Default color
                box_color = (128, 128, 128)

                # If the barcode matches and this is the box with the highest score among those that match
                # if draw_colored_boxes and i == closest_detection_index: 
                if i == highest_score_index: 
                    # rospy.loginfo(f"Matched barcode with highest score, drawing green box.")
                    box_color = (0, 255, 0)  # Green for the highest score among matching barcodes

                # Draw the box with the determined color
                annotator2.box_label(box.xyxy[0], label=f'{self.model.names[label_index]} {scores[i]:.2f}', color=box_color)

        # Convert the annotated frame to a ROS image message and publish
        frame_with_barcode = annotator2.result()
        raw_image_barcode = self.bridge.cv2_to_imgmsg(frame_with_barcode, encoding="bgr8")
        self.pub_img_barcode.publish(raw_image_barcode)


        # visualization
        #self.plot_detection_results(rotated_rgb_image, results)
        #self.show_rotated_results(rgb_image, boxes, angle)
        #cv2.waitKey(1)


import time

if __name__ == "__main__":
    rospy.init_node("product_detector")
    detector = ProductDetector()
    t0 = time.time()
    while not rospy.is_shutdown():
        detector.run()
        detector.rate.sleep()
        # print(f"product detection rate: {1/(time.time() - t0)}")
        t0 = time.time()
