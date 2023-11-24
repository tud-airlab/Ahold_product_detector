#!/usr/bin/env python3
import rospy
import tf
import numpy as np
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import cv2


class RotationCompensation:
    def __init__(self) -> None:
        self.listener = tf.TransformListener()
        rospy.sleep(1.0)

    def rotate_image(self, img, stamp):
        (trans, rot) = self.listener.lookupTransform(
            "base_link", "camera_color_optical_frame", stamp
        )
        _, self.phi, _ = euler_from_quaternion(rot)
        return rotate_image(img, -180 * self.phi / np.pi), self.phi

    def rotate_bounding_boxes(self, boxes_xywh, image, phi=0):
        if hasattr(self, "phi"):
            angle = 180 * self.phi / np.pi
        else:
            angle = 180 * phi / np.pi

        rotated_boxes_xywh = rotate_bounding_boxes(boxes_xywh, image, angle)

        if hasattr(self, "phi"):
            return rotated_boxes_xywh, self.phi
        else:
            return rotated_boxes_xywh, phi

    def rotate_point(self, xyz, image_shape=[720, 1280]):
        x, y, _ = xyz
        x = x-image_shape[1]/2
        y = y-image_shape[0]/2
        new_x = x * np.cos(self.phi) - y * np.sin(self.phi)
        new_y = x * np.sin(self.phi) + y * np.cos(self.phi)
        new_x = new_x+image_shape[1]/2
        new_y = new_y+image_shape[0]/2
        return np.array([new_x, new_y]) 


def rotate_bounding_boxes(boxes_xywh, image, angle):
    centers = boxes_xywh[:, 0:2]
    dimensions = boxes_xywh[:, 2:4]
    rot_mat = cv2.getRotationMatrix2D(
        (image.shape[1] / 2, image.shape[0] / 2), angle, 1
    )
    centers_rotated = (
        rot_mat @ np.concatenate((centers, np.ones((centers.shape[0], 1))), axis=1).T
    ).T[:, :2]
    return np.concatenate((centers_rotated, dimensions), axis=1)


def rotate_image(image, angle):
    """Rotates image, and fills background with black or returns the largest rectangle possible inside rotated image"""
    rotation_matrix = cv2.getRotationMatrix2D(
        (image.shape[1] / 2, image.shape[0] / 2), angle, 1
    )
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (image.shape[1], image.shape[0])
    )
    return rotated_image
