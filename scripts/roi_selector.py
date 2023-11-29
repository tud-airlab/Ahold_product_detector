import uuid
from pathlib import Path

import cv2
import rospy


class ROISelector:
    def __init__(self):
        self.y_end = None
        self.x_end = None
        self.x_start = None
        self.y_start = None
        self.click_count = 0
        self.reference_img = None
        self.window_name = "Select Class"

    def _select_roi(self, event, x, y, flags, param):
        image_temp = self.reference_img.copy()

        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_count += 1
            if self.click_count % 2 == 0:
                self.x_end, self.y_end = x, y
                cv2.rectangle(image_temp, (self.x_start, self.y_start), (x, y), (0, 0, 255), 2)
                cv2.imshow(self.window_name, image_temp)
            else:
                self.x_start, self.y_start = x, y
                self.x_end = None
                self.y_end = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.x_start is not None and self.y_start is not None:
                if self.x_end is not None and self.y_end is not None:
                    return
                else:
                    cv2.rectangle(image_temp, (self.x_start, self.y_start), (x, y), (0, 0, 255), 2)
                    cv2.imshow(self.window_name, image_temp)

    def get_cropped_img(self):
        if self.x_start is not None and self.x_end is not None and self.y_start is not None and self.y_end is not None:
            y_start, y_end = tuple(sorted((self.y_start, self.y_end)))
            x_start, x_end = tuple(sorted((self.x_start, self.x_end)))
            return self.reference_img[y_start:y_end, x_start:x_end]

    def __call__(self, image_with_detections, raw_img, class_name, path_to_dataset=None, vis_result=True):
        window_name_1 = "Visualization Result"
        window_name_2 = "Press space or enter to select image"
        if class_name is None and vis_result:
            cv2.imshow(window_name_1, image_with_detections)
            cv2.destroyWindow(window_name_2)
            cv2.waitKey(1)
            return None, None
        else:
            cv2.imshow(window_name_2, image_with_detections)
            cv2.destroyWindow(window_name_1)

            key = cv2.waitKey(1)
            if key != 13 and key != 32:  # Press enter or space to confirm this is the one you want
                return None, class_name
            cv2.destroyWindow(window_name_2)

            unset_class = None
            self.reference_img = raw_img
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self._select_roi)
            while True:
                cv2.imshow(self.window_name, self.reference_img)
                key = cv2.waitKey(0)

                if key == 27:
                    return None, class_name
                if self.get_cropped_img() is not None:
                    if key == 13 or key == 32:
                        unset_class = class_name
                        break  # Confirm selection, rerun the wizard because class is not set
                    elif key == ord('y'):
                        break  # y is forever ready

            cropped_img = self.get_cropped_img()
            cv2.imshow("ROI", cropped_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if path_to_dataset is None:
                output_directory = Path(__file__).parent.joinpath(class_name)
                rospy.logwarn(f"No path to dataset provided, output directory will be: {output_directory}")
            else:
                output_directory = path_to_dataset.joinpath(class_name)
                rospy.loginfo(f"Output will be written to: {output_directory}")
            output_directory.mkdir(exist_ok=True)
            cv2.imwrite(str(output_directory.joinpath(uuid.uuid4().hex + ".png")), cropped_img)
            return output_directory, unset_class
