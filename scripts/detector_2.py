#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Union

import PIL.Image
import cv2
import numpy as np
import rospy
import torch
import ultralytics
import tf
from ahold_product_detection.msg import Detection, RotatedBoundingBox
from ahold_product_detection.srv import *
from cv_bridge import CvBridge
# message and service imports
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from ultralytics.utils.plotting import Annotator

from rotation_compensation import RotationCompensation
from pmf_data_helpers import IMAGE_LOADER, ImageLoader, SEEN_COLOR, SEEN_CLASSES, UNSEEN_COLOR, DEFAULT_COLOR
from pmf_interface import PMF


class YoloHelper(ultralytics.YOLO):
    def __init__(self, yolo_weights_path, bounding_box_conf_threshold, device, image_loader: ImageLoader):
        super().__init__(yolo_weights_path)
        self._device = device
        self._image_loader = image_loader
        if 0 <= bounding_box_conf_threshold < 1:
            self.bounding_box_conf_threshold = bounding_box_conf_threshold
        else:
            raise Exception("No valid confidence threshold supplied")

    def predict(self, source=None, stream=False, predictor=None, **kwargs):
        prediction = \
            super().predict(source=source, stream=stream, predictor=predictor, device=self._device, **kwargs)[0]
        bounding_boxes = prediction.boxes[prediction.boxes.conf > self.bounding_box_conf_threshold]
        cropped_images = self._crop_img_with_bounding_boxes(source, bounding_boxes)
        return cropped_images, bounding_boxes

    def _crop_img_with_bounding_boxes(self, image: PIL.Image.Image, bounding_boxes: ultralytics.engine.results.Boxes):
        """
        Crop image with predicted bounding boxes
        """
        multi_image_tensor = torch.empty(
            size=(len(bounding_boxes), 3, self._image_loader.image_size, self._image_loader.image_size),
            dtype=torch.float, device="cuda:0",
            requires_grad=False)
        i = 0
        for cx, cy, width, height in bounding_boxes.xywh:
            cropped_image = image.crop(
                (int(cx - width / 2), int(cy - height / 2), int(cx + width / 2), int(cy + height / 2)))
            multi_image_tensor[i] = self._image_loader(cropped_image)
            i += 1
        return multi_image_tensor


class CameraData:
    def __init__(self, listen_to_pointcloud: bool = False) -> None:
        # Setup ros subscribers and service
        self.depth_subscriber = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.rgb_subscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)

        self.depth_msg = None
        self.rgb_msg = None
        self.pointcloud_msg = None

        rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image, timeout=10)
        rospy.wait_for_message("/camera/color/image_raw", Image, timeout=10)

        if listen_to_pointcloud:
            self.pointcloud_subscriber = rospy.Subscriber("/camera/depth/color/points", PointCloud2,
                                                          self.pointcloud_callback)
            rospy.wait_for_message("/camera/depth/color/points", PointCloud2, timeout=15)

    def depth_callback(self, data):
        self.depth_msg = data

    def pointcloud_callback(self, data):
        self.pointcloud_msg = data

    def rgb_callback(self, data):
        self.rgb_msg = data

    @property
    def data(self):
        # TODO: timesync or check if the time_stamps are not too far apart (acceptable error)
        return self.rgb_msg, self.depth_msg, self.pointcloud_msg, self.rgb_msg.header.stamp


class ProductDetector2:
    def __init__(self, yolo_weights_path: Path, pmf_weights_path: Path, visualize_results: bool = False,
                 yolo_conf_threshold: float = 0.2, pmf_conf_threshold: float = 0.5, device: str = "cpu",
                 dataset_path: Path = None, reload_prototypes: bool = False, image_loader: ImageLoader = IMAGE_LOADER,
                 debug_clf: bool = False, rotate: bool = True):

        self._barcode = None
        self.rotate = rotate
        self.visualize_results = visualize_results

        self.camera = CameraData()
        self.rotation_compensation = RotationCompensation()
        self.rate = rospy.Rate(30)
        self.bridge = CvBridge()
        self.tf_listener = tf.TransformListener()

        self.yolo = YoloHelper(yolo_weights_path, yolo_conf_threshold, device=device, image_loader=image_loader)
        self.classifier = PMF(pmf_weights_path, classification_confidence_threshold=pmf_conf_threshold,
                              image_loader=image_loader, path_to_dataset=dataset_path, device=device,
                              reload_prototypes=reload_prototypes)

        self.debug_clf = debug_clf

        self.class_sub = rospy.Subscriber("/detection_class", String, self.set_detection_class)
        self.detection_pub = rospy.Publisher("/detection_results", Detection, queue_size=10)
        self.visualization_pub = rospy.Publisher("/detection_image", Image, queue_size=10)
        self._check_barcode_timer = rospy.Timer(rospy.Duration(0.1), self.check_barcode_update)
        self.intrinsics_subscriber = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.intrinsics_callback)
        self._status_sub = rospy.Subscriber("/all_nodes_active", Bool, self.nodes_check_cb)
        self.all_nodes_up = False

        # Parameters from original detector
        self._currently_recording = False
        self._currently_playback = False
        self.start_tablet_subscribers()

    def nodes_check_cb(self, msg):
        self.all_nodes_up = msg.data

    def intrinsics_callback(self, data):
        self.intrinsics = np.array(data.K).reshape((3, 3))

    def start_tablet_subscribers(self):
        self._start_record_sub = rospy.Subscriber("/tablet/start_record", Bool, self.start_record_callback)
        self._stop_record_sub = rospy.Subscriber("/tablet/stop_record", Bool, self.stop_record_callback)
        self._playback_subscriber = rospy.Subscriber("/tablet/start_playback", Bool, self.start_playback_callback)
        self._stop_playback_subscriber = rospy.Subscriber("/tablet/stop_playback", Bool, self.stop_playback_callback)

    def start_playback_callback(self, msg):
        if msg.data:
            self._currently_playback = True

    def stop_playback_callback(self, msg):
        if msg.data:
            self._currently_playback = False

    def start_record_callback(self, msg):
        if msg.data:
            self._currently_recording = True
            self._barcode = rospy.get_param("/barcode")
            rospy.loginfo(f"Start recording with barcode: {self._barcode}")
            rospy.loginfo(f"Barcode type: {type(self._barcode)}")  # Add this line

    def stop_record_callback(self, msg):
        if msg.data:
            self._currently_recording = False

    def get_tracked_product_in_camera_image(self):
        try:
            xyz, _ = self.tf_listener.lookupTransform(
                "/camera_color_optical_frame", "/desired_product", rospy.Time(0)
            )
        except Exception as e:
            rospy.loginfo_throttle(10, f"No tracked product tf found, {e}")
            return None
        pixel_vector = self.intrinsics @ xyz
        pixel_vector = pixel_vector / pixel_vector[2]
        pixel_vector = self.rotation_compensation.rotate_point(pixel_vector)
        return pixel_vector[:2]

    def _plot_detection_results(self, frame: Image, bounding_boxes, scores, classes):
        """
        Plotting function for showing preliminary detection results for debugging
        """
        raw_image = np.ascontiguousarray(np.asarray(frame)[:, :, ::-1])
        annotator = Annotator(raw_image.copy(), font_size=6)
        labels = [f"{class_[:10]}... {score.item():.2f}" if score != 0 else f"{class_}" for class_, score in
                  zip(classes, scores)]
        # Color bounding boxes based on if they are seen/unseen
        colors = [SEEN_COLOR if class_ in SEEN_CLASSES else DEFAULT_COLOR if score == 0 else UNSEEN_COLOR for
                  class_, score in
                  zip(classes, scores)]

        for box, label, color in zip(bounding_boxes, labels, colors):
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            annotator.box_label(b, label, color)

        result = annotator.result()

        tracked_product_xy = self.get_tracked_product_in_camera_image()
        if tracked_product_xy is not None:
            cv2.circle(raw_image, np.array(tracked_product_xy).astype(int), radius=20, color=(255, 0, 0), thickness=-1)

        # Draw status circle
        if self.all_nodes_up:
            status_color = (0, 255, 0)  # colors in BGR
        else:
            status_color = (0, 0, 255)
        self.status_radius = 20
        cv2.circle(raw_image, np.array([1.5 * self.status_radius, 1.5 * self.status_radius]).astype(int),
                   radius=self.status_radius, color=status_color, thickness=-1)

        return result, raw_image

    @staticmethod
    def generate_detection_message(time_stamp, boxes, scores, labels, rgb_msg, depth_msg) -> Detection:
        detection_msg = Detection()
        detection_msg.header.stamp = time_stamp

        bboxes_list = []
        for bbox, label, score in zip(boxes, labels, scores):
            if score > 0:
                bbox_msg = RotatedBoundingBox()

                bbox_msg.x = int(bbox[0])
                bbox_msg.y = int(bbox[1])
                bbox_msg.w = int(bbox[2])
                bbox_msg.h = int(bbox[3])
                # bbox_msg.label = label
                bbox_msg.score = score

                bboxes_list.append(bbox_msg)

        detection_msg.detections = bboxes_list
        detection_msg.rgb_image = rgb_msg
        detection_msg.depth_image = depth_msg

        return detection_msg

    def check_barcode_update(self, _):
        new_barcode = rospy.get_param("/barcode", None)
        if new_barcode != self._barcode:
            self._barcode = new_barcode
            self.set_detection_class(str(new_barcode))

    def set_detection_class(self, class_to_find: Union[String, str]):
        rospy.loginfo("received request to change clas")
        class_to_find = class_to_find.data if isinstance(class_to_find, String) else class_to_find
        self.classifier.set_class_to_find(class_to_find)

    def run(self):
        if self.classifier.get_current_class() is not None:
            try:
                rgb_msg, depth_msg, pointcloud_msg, time_stamp = self.camera.data
            except Exception as e:
                rospy.logerr(f"Couldn't read camera data. Error: %s", e)
                return

            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            if self.rotate:
                rotated_image, angle = self.rotation_compensation.rotate_image(rgb_image, time_stamp)
                rotated_image = PIL.Image.fromarray(rotated_image[..., ::-1])  # Convert to PIL image
                cropped_images, bounding_boxes = self.yolo.predict(source=rotated_image, show=False, save=False,
                                                                   verbose=False, agnostic_nms=True)
                bounding_boxes_xywh, _ = self.rotation_compensation.rotate_bounding_boxes(bounding_boxes.xywh.cpu(),
                                                                                          rgb_image, angle)
                scores, labels = self.classifier(cropped_images, debug=self.debug_clf)

                result, raw_image = self._plot_detection_results(frame=rotated_image, bounding_boxes=bounding_boxes,
                                                                 scores=scores, classes=labels)
                if self.visualize_results:
                    self.visualization_pub.publish(self.bridge.cv2_to_imgmsg(result))
                self.classifier.check_for_new_class_selection(result, raw_image, vis_results=self.visualize_results)
                detection_results_msg = self.generate_detection_message(time_stamp=time_stamp,
                                                                        boxes=bounding_boxes_xywh,
                                                                        scores=scores, labels=labels, rgb_msg=rgb_msg,
                                                                        depth_msg=depth_msg)
                self.detection_pub.publish(detection_results_msg)
            else:
                rgb_image = PIL.Image.fromarray(rgb_image[..., ::-1])  # Convert to PIL image
                cropped_images, bounding_boxes = self.yolo.predict(source=rgb_image, show=False, save=False,
                                                                   verbose=False, agnostic_nms=True)
                scores, labels = self.classifier(cropped_images, debug=self.debug_clf)

                result, raw_image = self._plot_detection_results(frame=rgb_image, bounding_boxes=bounding_boxes,
                                                                 scores=scores,
                                                                 classes=labels)
                if self.visualize_results:
                    self.visualization_pub.publish(self.bridge.cv2_to_imgmsg(result))
                self.classifier.check_for_new_class_selection(result, raw_image, vis_results=self.visualize_results)
                detection_results_msg = self.generate_detection_message(time_stamp=time_stamp, boxes=bounding_boxes.xywh.cpu(),
                                                                        scores=scores, labels=labels, rgb_msg=rgb_msg,
                                                                        depth_msg=depth_msg)
                self.detection_pub.publish(detection_results_msg)


if __name__ == "__main__":
    rospy.init_node("product_detector_2")
    yolo_weights_path = Path(__file__).parent.parent.joinpath("models", "YOLO_just_products.pt")
    pmf_weights_path = Path(__file__).parent.parent.joinpath("models", "PMF.pth")
    DEBUG = False  # Flag for testing without robot attached
    if DEBUG:
        dataset_path = Path(__file__).parent.parent.joinpath("data", "Custom-Set_FULL")
        detector = ProductDetector2(rotate=False,
                                    yolo_weights_path=yolo_weights_path,
                                    yolo_conf_threshold=0.2,
                                    pmf_weights_path=pmf_weights_path,
                                    pmf_conf_threshold=0.65,
                                    dataset_path=dataset_path,
                                    device="cuda:0",
                                    visualize_results=True,
                                    reload_prototypes=False,
                                    debug_clf=False)
        detector.classifier.set_class_to_find("6_AH_Hollandse_Bruine_Bonen - 8710400035862")
    else:
        detector = ProductDetector2(yolo_weights_path=yolo_weights_path,
                                    yolo_conf_threshold=0.2,
                                    pmf_weights_path=pmf_weights_path,
                                    pmf_conf_threshold=0.70,
                                    device="cuda:0",
                                    visualize_results=True)
    while not rospy.is_shutdown():
        try:
            detector.run()
        except Exception as e:
            rospy.logerr(f"Couldn't run detection because of: {e}")
        detector.rate.sleep()
