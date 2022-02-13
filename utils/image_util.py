import math
import cv2
import numpy as np
from PIL import Image


def get_image(path):
    return cv2.imread(path)


def resize_image(image, size=(28, 28)):
    return cv2.resize(image, size)


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert(image):
    return ~image


def convert_from_png_to_jpg(png_path, jpg_path):
    image = get_image(png_path)
    cv2.imwrite(jpg_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return jpg_path


def convert_from_gif_to_jpg(gif_path, jpg_path):
    image = Image.open(gif_path)
    image.save(jpg_path)


def convert_from_gif_to_jpg_without_saving(gif_path):
    image = Image.open(gif_path).convert('RGB')
    image = np.array(image)
    return image[:, :, ::-1].copy()


def enhance(image):
    cv2.fastNlMeansDenoising(image, image, 5.0, 7, 21)
    image = np.where(image <= 32, 0, image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.dilate(image, kernel, iterations=1)
    min_val, max_val = np.min(image), np.max(image)
    diff = max_val - min_val
    if diff > 0:
        image = 255 * ((image - min_val) / diff)
    return image


def trim(image):
    blur = cv2.blur(image, (5, 5))
    gray = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)[1]
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    if w >= 1 and h >= 1:
        trimmed = image[y:y + h, x:x + w]
        top_bottom_border = int(0.1 * h)
        left_right_border = int(0.1 * w)
        bordered = cv2.copyMakeBorder(
            trimmed,
            top=top_bottom_border,
            bottom=top_bottom_border,
            left=left_right_border,
            right=left_right_border,
            borderType=cv2.BORDER_CONSTANT,
            value=255
        )
        return bordered
    return image


def crop_table(image, top_x, top_y, btm_x, btm_y):
    image_crop = image[top_y:btm_y, top_x:btm_x]
    return image_crop


def preprocess(image, size=(28, 28), trim_flag=True):
    if len(image.shape) == 3:
        image = grayscale(image)
    if trim_flag:
        image = trim(image)
    image = resize_image(image, size)
    image = convert(image)
    image = enhance(image)
    return image
