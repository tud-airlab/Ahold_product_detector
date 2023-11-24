#!/usr/bin/env python3
from sensor_msgs.msg import CameraInfo
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
        self._status_sub = rospy.Subscriber("/all_nodes_active", Bool, self.nodes_check_cb)
        self.tf_listener = tf.TransformListener()
        self._trigger_detection = rospy.Subscriber("/product_detector/trigger", Bool, self.trigger_detection)
        self._currently_detecting = False

        self.bridge = CvBridge()

        self._currently_recording = False
        self._currently_playback = False
        self.all_nodes_up = False
        self.start_tablet_subscribers()
        self._barcode = None
        self._check_barcode_timer = rospy.Timer(rospy.Duration(0.1), self.check_barcode_update)

        self._assigned_detection_subscriber = rospy.Subscriber('/product_tracker/assigned_detection_image_coordinates',
                                                               Float32MultiArray, self.assigned_detection_cb)
        self.intrinsics_subscriber = rospy.Subscriber(
            "/camera/color/camera_info", CameraInfo, self.intrinsics_callback
        )
        self._assigned_detection_uv = [-1, -1]
        self.status_radius = 20

    def intrinsics_callback(self, data):
        self.intrinsics = np.array(data.K).reshape((3, 3))

    def trigger_detection(self, msg):
        self._currently_detecting = msg.data

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

    def nodes_check_cb(self, msg):
        self.all_nodes_up = msg.data

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

    def get_tracked_product_in_camera_image(self):
        try:
            xyz, _ = self.tf_listener.lookupTransform(
                "/camera_color_optical_frame", "/desired_product", rospy.Time(0)
            )
        except Exception as e:
            # rospy.loginfo(f"No tracked product tf found, {e}")
            return None
        pixel_vector = self.intrinsics @ xyz
        pixel_vector = pixel_vector / pixel_vector[2]
        pixel_vector = self.rotation_compensation.rotate_point(pixel_vector)
        return pixel_vector[:2]

    def run(self):
        try:
            rgb_msg, depth_msg, pointcloud_msg, time_stamp = self.camera.data
        except Exception as e:
            rospy.logerr(f"Couldn't read camera data: {e}")
            return

        # rotate input
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        try:
            rotated_rgb_image, _ = self.rotation_compensation.rotate_image(
                rgb_image, time_stamp
            )
        except Exception as e:
            rospy.logerr(f"Couldn't rotate the image: {e}")
            return

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
            conf=0.4
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
        # draw_colored_boxes = self._currently_recording and self._barcode is not None
        # draw_colored_boxes = True
        tracked_product_xy = self.get_tracked_product_in_camera_image()
        for r in results:
            annotator2 = Annotator(rotated_rgb_image)
            boxes = r.boxes
            labels = r.boxes.cls.cpu().numpy()

            if self._barcode == None or tracked_product_xy is None:
                tracked_product_index = -1
            else:
                try:
                    yolo_id = rospy.get_param("requested_yolo_id")
                except Exception as e:
                    yolo_id = -1
                    rospy.logwarn(e)

                # Filter boxes that match the barcode
                matching_boxes_idxs = [i for i, label in enumerate(labels) if label == yolo_id]
                if len(matching_boxes_idxs) != 0:
                    matching_boxes_xy = np.array([box.xywh.cpu().numpy()[0][:2] for box in boxes[matching_boxes_idxs]])
                    dists = np.linalg.norm(matching_boxes_xy - tracked_product_xy, axis=1)
                    closest_match_idx = np.argmin(dists)
                    tracked_product_index = matching_boxes_idxs[closest_match_idx]
                else:
                    tracked_product_index = -1

                # # Find the highest score among the matching boxes
                # if matching_boxes_indices:
                #     highest_score_index = matching_boxes_indices[np.argmax(scores[matching_boxes_indices])]

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
                if i == tracked_product_index:
                    continue

                # Draw the box with the determined color
                annotator2.box_label(box.xyxy[0], label=f'{self.model.names[label_index]} {scores[i]:.2f}',
                                     color=box_color)

            if tracked_product_index != -1:
                label_index = int(labels[tracked_product_index])  # Convert label to int
                box_color = (0, 255, 0)  # Green for the highest score among matching barcodes
                # Draw the box with the determined color
                annotator2.box_label(boxes[tracked_product_index].xyxy[0],
                                     label=f'{self.model.names[label_index]} {scores[tracked_product_index]:.2f}',
                                     color=box_color)

        # Convert the annotated frame to a ROS image message and publish
        frame_with_barcode = annotator2.result()
        if tracked_product_xy is not None:
            cv2.circle(frame_with_barcode, np.array(tracked_product_xy).astype(int), radius=20, color=(255, 0, 0),
                       thickness=-1)

        # Draw status circle
        if self.all_nodes_up:
            status_color = (0, 255, 0)  # colors in BGR
        else:
            status_color = (0, 0, 255)

        cv2.circle(frame_with_barcode, np.array([1.5 * self.status_radius, 1.5 * self.status_radius]).astype(int),
                   radius=self.status_radius, color=status_color, thickness=-1)

        raw_image_barcode = self.bridge.cv2_to_imgmsg(frame_with_barcode, encoding="bgr8")
        self.pub_img_barcode.publish(raw_image_barcode)

        # visualization
        # self.plot_detection_results(rotated_rgb_image, results)
        # self.show_rotated_results(rgb_image, boxes, angle)
        # cv2.waitKey(1)


import time

if __name__ == "__main__":
    rospy.init_node("product_detector")
    detector = ProductDetector()
    t0 = time.time()
    while not rospy.is_shutdown():
        # if detector._currently_detecting:
        detector.run()
        detector.rate.sleep()
        # print(f"product detection rate: {1/(time.time() - t0)}")
        t0 = time.time()
