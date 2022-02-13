import cv2
import numpy as np


class TableDetection:

    def detect_table(self, image, questions):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(gray, (5, 5))
        thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)[1]

        kernel_h = np.ones((1, 100), np.uint8)
        kernel_v = np.ones((250, 1), np.uint8)
        image_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
        image_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
        image_union = image_h | image_v

        kernel = np.ones((50, 50), np.uint8)
        image_final = cv2.morphologyEx(image_union, cv2.MORPH_CLOSE, kernel,
                                       iterations=3)
        contours, hierarchy = cv2.findContours(image_final,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        height, width = image.shape[:2]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = (h / questions) / w
            if (width * 0.1) < w < (width * 0.2) and 0.3 > ratio > 0.2:
                return image[y - int(0.005 * h): y + int(h * 1.005),
                       x - int(0.025 * w): x + int(1.025 * w)]
