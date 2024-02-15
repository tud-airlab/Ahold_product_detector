#!/usr/bin/env python3
import os
from pathlib import Path
import PIL.Image
import rospy
import torch
import numpy as np
import cv2
import ultralytics
from ahold_product_detection.msg import Detection, RotatedBoundingBox
from ahold_product_detection.srv import (
    AddClass,
    AddClassResponse,
    GetCroppedImages,
    GetCroppedImagesResponse,
    GetClassNames,
    GetClassNamesResponse,
    SetDetectionClass,
    SetDetectionClassResponse,
)
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from ultralytics.utils.plotting import Annotator
from rotation_compensation import RotationCompensation
from pmf_data_helpers import (
    IMAGE_LOADER,
    ImageLoader,
    SEEN_COLOR,
    SEEN_CLASSES,
    UNSEEN_COLOR,
    DEFAULT_COLOR,
)
from copy import deepcopy, copy
from pmf_interface import PMF

SHOW = os.getenv("SHOW")


def generate_detection_message(
    time_stamp, boxes, scores, labels, rgb_msg, depth_msg
) -> Detection:
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


def plot_detection_results(frame: Image, bounding_boxes, scores, classes):
    """
    Plotting function for showing preliminary detection results for debugging
    """
    raw_image = np.ascontiguousarray(np.asarray(frame)[:, :, ::-1])
    annotator = Annotator(raw_image.copy(), font_size=6)
    labels = [
        f"{class_[:10]}... {score.item():.2f}" if score != 0 else f"{class_}"
        for class_, score in zip(classes, scores)
    ]
    # Color bounding boxes based on if they are seen/unseen
    colors = [
        SEEN_COLOR
        if class_ in SEEN_CLASSES
        else DEFAULT_COLOR
        if score == 0
        else UNSEEN_COLOR
        for class_, score in zip(classes, scores)
    ]

    for box, label, color in zip(bounding_boxes, labels, colors):
        b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
        annotator.box_label(b, label, color)

    result = annotator.result()
    return result, raw_image


class YoloHelper(ultralytics.YOLO):
    def __init__(
        self,
        yolo_weights_path,
        bounding_box_conf_threshold,
        device,
        image_loader: ImageLoader,
    ):
        super().__init__(yolo_weights_path)
        self._device = device
        self._image_loader = image_loader
        if 0 <= bounding_box_conf_threshold < 1:
            self.bounding_box_conf_threshold = bounding_box_conf_threshold
        else:
            raise Exception("No valid confidence threshold supplied")

    def predict(self, source=None, stream=False, predictor=None, **kwargs):
        prediction = super().predict(
            source=source,
            stream=stream,
            predictor=predictor,
            device=self._device,
            verbose=False,
            **kwargs,
        )[0]
        bounding_boxes = prediction.boxes[
            prediction.boxes.conf > self.bounding_box_conf_threshold
        ]
        cropped_images, images_pmf = self._crop_img_with_bounding_boxes(
            source, bounding_boxes
        )
        return cropped_images, images_pmf, bounding_boxes

    def _crop_img_with_bounding_boxes(
        self, image: PIL.Image.Image, bounding_boxes: ultralytics.engine.results.Boxes
    ):
        """
        Crop image with predicted bounding boxes
        """
        multi_image_tensor = torch.empty(
            size=(
                len(bounding_boxes),
                3,
                self._image_loader.image_size,
                self._image_loader.image_size,
            ),
            dtype=torch.float,
            device="cuda:0",
            requires_grad=False,
        )
        i = 0
        cropped_images = []
        for cx, cy, width, height in bounding_boxes.xywh:
            cropped_image = image.crop(
                (
                    int(cx - width / 2),
                    int(cy - height / 2),
                    int(cx + width / 2),
                    int(cy + height / 2),
                )
            )
            cropped_images.append(cropped_image)
            multi_image_tensor[i] = self._image_loader(cropped_image)
            i += 1
        return cropped_images, multi_image_tensor


class CameraData:
    def __init__(self, listen_to_pointcloud: bool = False) -> None:
        # Setup ros subscribers and service
        self.depth_subscriber = rospy.Subscriber(
            "/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback
        )
        self.rgb_subscriber = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.rgb_callback
        )

        self.depth_msg = None
        self.rgb_msg = None
        self.pointcloud_msg = None

        rospy.wait_for_message(
            "/camera/aligned_depth_to_color/image_raw", Image, timeout=10
        )
        rospy.wait_for_message("/camera/color/image_raw", Image, timeout=10)

        if listen_to_pointcloud:
            self.pointcloud_subscriber = rospy.Subscriber(
                "/camera/depth/color/points", PointCloud2, self.pointcloud_callback
            )
            rospy.wait_for_message(
                "/camera/depth/color/points", PointCloud2, timeout=15
            )

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
    def __init__(self, yolo_weights_path, pmf_weights_path) -> None:
        self.bridge = CvBridge()
        self.rate = rospy.Rate(30)
        self.rotation_compensation = RotationCompensation()
        self.camera = CameraData()
        self.result_publisher = rospy.Publisher(
            "/product_detector/result_image/compressed", CompressedImage, queue_size=1
        )
        self.detection_pub = rospy.Publisher(
            "/detection_results", Detection, queue_size=10
        )
        self.curent_class_publisher = rospy.Publisher(
            "/product_detector/current_class", String
        )
        self.add_class_service = rospy.Service(
            "/product_detector/add_class", AddClass, self.add_class
        )
        self.get_class_names_service = rospy.Service(
            "/product_detector/get_class_names", GetClassNames, self.get_class_names
        )
        self.set_detection_class_service = rospy.Service(
            "/product_detector/set_detection_class",
            SetDetectionClass,
            self.set_detection_class,
        )
        self.get_cropped_images_service = rospy.Service(
            "/product_detector/get_cropped_images",
            GetCroppedImages,
            self.provide_cropped_images,
        )
        self.yolo = YoloHelper(
            yolo_weights_path,
            bounding_box_conf_threshold=0.6,
            device="cuda:0",
            image_loader=IMAGE_LOADER,
        )
        self.classifier = PMF(
            pmf_weights_path,
            classification_confidence_threshold=0.8,
            image_loader=IMAGE_LOADER,
            device="cuda:0",
        )

    def set_detection_class(self, req):
        class_name = req.class_name.strip()
        (
            class_list,
            class_prototypes,
        ) = self.classifier.prototype_loader._load_prototypes_from_dict(
            class_name, self.classifier.amount_of_prototypes
        )
        self.classifier.prototype_loader.classes = class_list
        self.class_list = class_list
        self.classifier.protonet.update_prototypes(class_prototypes)
        return SetDetectionClassResponse(success=True)

    def get_class_names(self, req):
        class_names = self.classifier.prototype_loader.prototype_dict.keys()
        return GetClassNamesResponse(class_names=class_names)

    def provide_cropped_images(self, req):
        cropped_image_msg_list = []
        for cropped_image in self.cropped_images:
            # cropped_image = np.transpose(np.array(cropped_image), (1, 2, 0))  # Now the shape is (80, 80, 3)
            cropped_image = np.array(cropped_image)
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_image_msg = self.bridge.cv2_to_compressed_imgmsg(cropped_image)
            cropped_image_msg_list.append(cropped_image_msg)
        return GetCroppedImagesResponse(cropped_images=cropped_image_msg_list)

    def add_class(self, req):
        cropped_images = []
        for img_msg in req.cropped_images:
            img = self.bridge.compressed_imgmsg_to_cv2(img_msg)
            cropped_images.append(img)
        self.classifier.add_class(req.name, cropped_images)
        return AddClassResponse(success=True)

    def run(self):
        if self.classifier.get_current_class() is None:
            return
        else:
            # Step 0: Get camera data
            rgb_msg, depth_msg, pointcloud_msg, time_stamp = self.camera.data
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            rotated_image, angle = self.rotation_compensation.rotate_image(
                rgb_image, time_stamp
            )
            rotated_image = PIL.Image.fromarray(
                rotated_image[..., ::-1]
            )  # Convert to PIL image

            # Step 1: Detect product bounding box proposals with yolo
            self.cropped_images, images_pmf, bounding_boxes = self.yolo.predict(
                source=rotated_image, agnostic_nms=True
            )

            # Step 2: Binary classification according to currently requested class
            scores, labels = self.classifier(images_pmf)

            # Step 3: Generate detection message for pose estimation and tracking
            detection_results_msg = generate_detection_message(
                time_stamp=time_stamp,
                boxes=bounding_boxes.xywh.cpu(),
                scores=scores,
                labels=labels,
                rgb_msg=rgb_msg,
                depth_msg=depth_msg,
            )
            self.detection_pub.publish(detection_results_msg)

            # Step 4: Send or show result
            result, raw_image = plot_detection_results(
                frame=rotated_image,
                bounding_boxes=bounding_boxes,
                scores=scores,
                classes=labels,
            )

            res = self.bridge.cv2_to_compressed_imgmsg(result)
            self.result_publisher.publish(res)
            self.curent_class_publisher.publish(
                String(data=self.classifier.current_class)
            )

            if SHOW:
                cv2.imshow("output", result)
                cv2.waitKey(1)


if __name__ == "__main__":
    rospy.init_node("product_detector_2")
    yolo_weights_path = Path(__file__).parent.parent.joinpath(
        "models", "YOLO_just_products.pt"
    )
    pmf_weights_path = Path(__file__).parent.parent.joinpath("models", "PMF_fuller.pth")

    detector = ProductDetector(yolo_weights_path, pmf_weights_path)
    detector.classifier.set_class_to_find("test")

    while not rospy.is_shutdown():
        try:
            detector.run()
        except Exception as e:
            rospy.logerr(f"Couldn't run detection because of: {e}")
        detector.rate.sleep()
