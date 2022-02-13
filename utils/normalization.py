import math
import os

import cv2
import numpy as np

from utils import image_util, exam_util


def rotate_matrix(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def detect_markers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 15)
    thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(~thresh,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    h, w, _ = image.shape
    markers = get_markers_from_contours(contours, w, h)
    return markers


def get_markers_from_contours(contours, width, height):
    markers = []
    h_upper, h_lower = 0.02 * height, 0.0055 * height
    w_upper, w_lower = 0.0275 * width, 0.008 * width
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        k = h / w
        if 0.85 < k < 1.15 and h_upper > h > h_lower and w_upper > w > w_lower:
            epsilon = 0.05 * cv2.arcLength(contours[i], True)
            approx = cv2.approxPolyDP(contours[i], epsilon, True)
            if len(approx) == 4:
                markers.append((x, y, w, h))
    return markers


def sort_markers(markers, center_x, center_y):
    markers = sorted(markers, key=lambda m: m[1])
    if len(markers) >= 4:
        if markers[0][0] < markers[1][0]:
            upper_left, upper_right = markers[:2]
        else:
            upper_right, upper_left = markers[:2]

        if markers[-1][0] < markers[-2][0]:
            lower_right, lower_left = markers[-2:]
        else:
            lower_left, lower_right = markers[-2:]
    else:
        upper_left, upper_right, lower_left, lower_right = (
            None, None, None, None)

        for point in markers:
            if point[0] < center_x and point[1] < center_y:
                upper_left = point
            elif point[0] < center_x and point[1] > center_y:
                lower_left = point
            elif point[0] > center_x and point[1] < center_y:
                upper_right = point
            elif point[0] > center_x and point[1] > center_y:
                lower_right = point

    if upper_left is not None and (
            upper_left[0] > center_x or upper_left[1] > center_y):
        upper_left = None
    if upper_right is not None and (
            upper_right[0] < center_x or upper_right[1] > center_y):
        upper_right = None
    if lower_left is not None and (
            lower_left[0] > center_x or lower_left[1] < center_y):
        lower_left = None
    if lower_right is not None and (
            lower_right[0] < center_x or lower_right[1] < center_y):
        lower_right = None

    markers_new = [upper_left, upper_right, lower_left, lower_right]
    if sum(x is None for x in markers_new) == 0:
        upper_left_distance = sum(upper_left[:2]) + sum(upper_left[2:]) / 2
        upper_right_distance = (center_x * 2 - upper_right[0] + upper_right[1]
                                + sum(upper_right[2:]) / 2)
        lower_left_distance = (lower_left[0] + center_y * 2 - lower_left[1] +
                               sum(lower_left[2:]) / 2)
        lower_right_distance = (center_x * 2 - lower_right[0] + center_y * 2 -
                                lower_right[1] + sum(lower_right[2:]) / 2)
        max_distance = max(upper_left_distance, upper_right_distance,
                           lower_left_distance, lower_right_distance)
        if upper_left_distance == max_distance:
            upper_left = None
        elif upper_right_distance == max_distance:
            upper_right = None
        elif lower_left_distance == max_distance:
            lower_left = None
        else:
            lower_right = None

    return upper_left, upper_right, lower_left, lower_right


def calculate_angle(point1, point2):
    x = point2[0] - point1[0]
    y = point2[1] - point1[1]
    if x == 0.:
        return 0
    return math.degrees(math.atan(y / x))


def get_marker_center(point):
    return point[0] + point[2] // 2, point[1] + point[3] // 2


def get_inner_markers_vertices(upper_left, upper_right, lower_left,
                               lower_right):
    upper_left_vertex = None
    upper_right_vertex = None
    lower_right_vertex = None
    lower_left_vertex = None

    if upper_left is not None:
        upper_left_vertex = [upper_left[0] + upper_left[2],
                             upper_left[1] + upper_left[3]]
    if upper_right is not None:
        upper_right_vertex = [upper_right[0],
                              upper_right[1] + upper_right[3]]
    if lower_left is not None:
        lower_left_vertex = [lower_left[0] + lower_left[2], lower_left[1]]
    if lower_right is not None:
        lower_right_vertex = [lower_right[0], lower_right[1]]

    if upper_left_vertex is None:
        upper_left_vertex = [
            upper_right_vertex[0] - (
                    lower_right_vertex[0] - lower_left_vertex[0]),
            lower_left_vertex[1] - (
                    lower_right_vertex[1] - upper_right_vertex[1])
        ]
    if upper_right_vertex is None:
        upper_right_vertex = [
            upper_left_vertex[0] + (
                    lower_right_vertex[0] - lower_left_vertex[0]),
            lower_right_vertex[1] - (
                    lower_left_vertex[1] - upper_left_vertex[1])
        ]
    if lower_right_vertex is None:
        lower_right_vertex = [
            lower_left_vertex[0] + (
                    upper_right_vertex[0] - upper_left_vertex[0]),
            upper_right_vertex[1] + (
                    lower_left_vertex[1] - upper_left_vertex[1])
        ]
    if lower_left_vertex is None:
        lower_left_vertex = [
            lower_right_vertex[0] - (
                    upper_right_vertex[0] - upper_left_vertex[0]),
            upper_left_vertex[1] + (
                    lower_right_vertex[1] - upper_right_vertex[1])
        ]

    return [upper_left_vertex, upper_right_vertex, lower_left_vertex,
            lower_right_vertex]


def get_box_points(rectangle):
    box = cv2.boxPoints(rectangle)
    box = np.int0(np.round(box))
    return box


def get_rectangle_from_markers(vertices):
    vertices = np.array([vertex for vertex in vertices if vertex is not None])
    rectangle = cv2.minAreaRect(vertices)
    return rectangle


def get_rectangle_dimensions(rectangle):
    if rectangle[1][0] > rectangle[1][1]:
        height, width = rectangle[1]
    else:
        width, height = rectangle[1]

    return int(width), int(height)


def draw_markers(image, markers, color=(0, 0, 255)):
    for marker in markers:
        cv2.rectangle(image, (marker[0], marker[1]),
                      (marker[0] + marker[2], marker[1] + marker[3]), color, 3)
    return image


def normalize_exam(image):
    height, width = image.shape[:2]
    markers = detect_markers(image)
    upper_left, upper_right, lower_left, lower_right = sort_markers(
        markers, width // 2, height // 2)
    inner_vertices = get_inner_markers_vertices(
        upper_left, upper_right, lower_left, lower_right)

    rectangle = get_rectangle_from_markers(inner_vertices)
    width, height = get_rectangle_dimensions(rectangle)
    box = get_box_points(rectangle)

    first, second = zip(*box)
    avg_x, avg_y = sum(first) / 4, sum(second) / 4

    src_pts = box.astype("float32")
    dst_pts = np.array([[0 if b[0] < avg_x else width,
                         0 if b[1] < avg_y else height] for b in box],
                       dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    normalized = cv2.warpPerspective(image, M, (width, height))
    return normalized


def normalize_exams_with_saving(path, new_path):
    exam_util.create_dir_if_needed(new_path)
    for exam_dir in os.listdir(path):
        exam_dir_path = os.path.join(path, exam_dir)
        new_exam_dir_path = os.path.join(new_path, exam_dir)
        exam_util.create_dir_if_needed(new_exam_dir_path)
        print(f'Starting {exam_dir}')
        for file in os.listdir(exam_dir_path):
            try:
                file_path = os.path.join(exam_dir_path, file)
                if file.endswith('.gif'):
                    image = image_util.convert_from_gif_to_jpg_without_saving(
                        file_path)
                else:
                    image = cv2.imread(file_path)
                normalized = normalize_exam(image)
                new_file_path = os.path.join(new_exam_dir_path,
                                             str(file.split('.')[0]) + '.jpg')
                cv2.imwrite(new_file_path, normalized)
            except:
                print(os.path.join(exam_dir_path, file))
        print(f'Finished {exam_dir}')


def normalize_exams(path):
    for exam_dir in os.listdir(path):
        exam_dir_path = os.path.join(path, exam_dir)
        print(f'Starting {exam_dir}')
        for file in os.listdir(exam_dir_path):
            try:
                file_path = os.path.join(exam_dir_path, file)
                if file.endswith('.gif'):
                    image = image_util.convert_from_gif_to_jpg_without_saving(
                        file_path)
                else:
                    image = cv2.imread(file_path)
                normalize_exam(image)
            except:
                print(os.path.join(exam_dir_path, file))
        print(f'Finished {exam_dir}')


def normalize_exams_dir(path):
    print(f'Starting {path}')
    for file in os.listdir(path):
        try:
            file_path = os.path.join(path, file)
            print(file)
            if file.endswith('.gif'):
                image = image_util.convert_from_gif_to_jpg_without_saving(
                    file_path)
            else:
                image = cv2.imread(file_path)
            normalize_exam(image)
        except:
            print(os.path.join(path, file))
    print(f'Finished {path}')
