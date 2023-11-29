from pathlib import Path

import cv2


class ROISelector:
    def __init__(self):
        self.y_end = None
        self.x_end = None
        self.x_start = None
        self.y_start = None
        self.click_count = 0
        self.reference_img = None
        self.window_name = "Select Class"

    def select_roi(self, event, x, y, flags, param):
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

    def annotate_img(self, input_img):
        self.reference_img = input_img

        cv2.imshow("Press space or enter to select image", self.reference_img)
        key = cv2.waitKey(50000)  # Should be 1 eventually
        cv2.destroyWindow("Press space or enter to select image")
        if key != 13 and key != 32:  # Press enter or space to confirm this is the one you want
            return

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.select_roi)
        while True:
            cv2.imshow(self.window_name, self.reference_img)
            key = cv2.waitKey(0)

            if key == 27:
                return
            elif self.x_start is not None and self.x_end is not None and self.y_start is not None and self.y_end is not None:
                if key == 13 or key == 32:  # Why twice *feature*??
                    break  # Confirm selection
                elif key == ord('y'):
                    break  # y is forever ready

        cropped_img = self.reference_img[self.y_start:self.y_end, self.x_start:self.x_end]
        cv2.imshow("ROI", cropped_img)
        cv2.imwrite("Roi.png", cropped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


selROI = ROISelector()
selROI.annotate_img()
