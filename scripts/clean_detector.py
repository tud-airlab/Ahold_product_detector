#!/usr/bin/env python3
import os
from pathlib import Path
import PIL.Image
import rospy
import torch
import numpy as np
import cv2
import ultralytics
from ahold_product_detection.srv import *
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
from copy import deepcopy
from pmf_interface import PMF

SHOW = os.getenv("SHOW")


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
            **kwargs,
        )[0]
        bounding_boxes = prediction.boxes[
            prediction.boxes.conf > self.bounding_box_conf_threshold
        ]
        cropped_images = self._crop_img_with_bounding_boxes(source, bounding_boxes)
        return cropped_images, bounding_boxes

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
        for cx, cy, width, height in bounding_boxes.xywh:
            cropped_image = image.crop(
                (
                    int(cx - width / 2),
                    int(cy - height / 2),
                    int(cx + width / 2),
                    int(cy + height / 2),
                )
            )
            multi_image_tensor[i] = self._image_loader(cropped_image)
            i += 1
        return multi_image_tensor


class ProductDetector:
    def __init__(self, yolo_weights_path, pmf_weights_path) -> None:
        self.bridge = CvBridge()
        self.rate = rospy.Rate(30)
        self.rotation_compensation = RotationCompensation()
        self.rgb_subscriber = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.rgb_callback
        )
        self.result_publisher = rospy.Publisher(
            "/product_detector/result_image", CompressedImage
        )
        self.curent_class_publisher = rospy.Publisher(
            "/product_detector/current_class", String
        )
        self.add_class_service = rospy.Service(
            "/product_detector/add_class",
            AddClass,
            self.add_class 
        )
        self.get_cropped_images_service = rospy.Service(
            "/product_detector/get_cropped_images",
            GetCroppedImages,
            self.provide_cropped_images 
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

    # workflow:
    # 1. request cropped images using service
    # 2. select the images on the user interface
    # 3. send the selected images to the add_class service. This service will also change the current class
    
    def provide_cropped_images(self, req):
        return GetCroppedImagesResponse(cropped_images=self.cropped_images)

    def add_class(self, req):
        print(req)
        return AddClassResponse(success=True)

    def rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        rotated_image, self.angle = self.rotation_compensation.rotate_image(
            self.rgb_image, msg.header.stamp
        )
        self.rotated_image = PIL.Image.fromarray(
            rotated_image[..., ::-1]
        )  # Convert to PIL image

    def run(self):
        if self.classifier.get_current_class() is None:
            return
        else:
            # Step 1: Detect product bounding box proposals with yolo
            self.cropped_images, bounding_boxes = self.yolo.predict(
                source=self.rotated_image, agnostic_nms=True
            )

            # Step 2: Binary classification according to currently requested class
            scores, labels = self.classifier(self.cropped_images)

            # Step 3: Send or show result
            result, raw_image = plot_detection_results(
                frame=self.rotated_image,
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
                cv2.imshow(result)
                cv2.wait_key(0)


if __name__ == "__main__":
    rospy.init_node("product_detector_2")
    yolo_weights_path = Path(__file__).parent.parent.joinpath(
        "models", "YOLO_just_products.pt"
    )
    pmf_weights_path = Path(__file__).parent.parent.joinpath("models", "PMF_fuller.pth")

    detector = ProductDetector(yolo_weights_path, pmf_weights_path)
    detector.classifier.set_class_to_find("6_AH_Hollandse_Bruine_Bonen - 8710400035862")

    while not rospy.is_shutdown():
        try:
            print("running detection")
            detector.run()
        except Exception as e:
            rospy.logerr(f"Couldn't run detection because of: {e}")
        detector.rate.sleep()
