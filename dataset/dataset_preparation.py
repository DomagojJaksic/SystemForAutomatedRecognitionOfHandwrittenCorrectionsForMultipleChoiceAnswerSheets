import os
import random

import cv2
import numpy as np
import torchvision

from dataset.dataset import rotate_minus_sign_dataset, transform
from table_detection import table_detection, table_extraction
from utils import exam_util, image_util, normalization


def prepare_empty_cells(source_path, destination_path):
    detection = table_detection.TableDetection()
    extraction = table_extraction.TableExtraction()
    for exam_dir in os.listdir(source_path):
        questions = 25 if exam_dir.startswith('MI') else 26
        exam_dir_path = os.path.join(source_path, exam_dir)
        new_exam_dir_path = os.path.join(destination_path, exam_dir)
        exam_util.create_dir_if_needed(new_exam_dir_path)

        print(f'Starting {exam_dir}')
        for file in os.listdir(exam_dir_path):
            try:
                file_path = os.path.join(exam_dir_path, file)
                image = image_util.convert_from_gif_to_jpg_without_saving(
                    file_path)
                image = normalization.normalize_exam(image)
                image = detection.detect_table(image, questions)
                cells = extraction.extract_table(image)

                i = 0
                for row in cells:
                    for cell in row:
                        x, y, w, h = cell[0]
                        cropped = image[y:y + h, x:x + w]
                        processed = image_util.preprocess(cropped, True)
                        r = random.uniform(0, 1)
                        if r < 0.25:
                            _add_circle_noise(processed)
                        elif 0.5 > r > 0.25:
                            _add_rectangle_noise(processed)
                        elif r > 0.75:
                            _add_point_noise(processed)

                        new_file_path = os.path.join(
                            new_exam_dir_path, f'{file.split(".")[0]}_{i}.jpg')
                        cv2.imwrite(new_file_path, processed)
                        i += 1
            except Exception as e:
                print(e)
                print(os.path.join(exam_dir_path, file))
        print(f'Finished {exam_dir}')


def _add_circle_noise(image):
    x_rand = random.randint(0, 27)
    y_rand = random.randint(0, 27)
    radius = random.randint(1, 3)
    color = random.randint(64, 255)
    cv2.circle(image, (x_rand, y_rand), radius=radius, color=color,
               thickness=-1)


def _add_rectangle_noise(image):
    x1, y1 = (random.randint(0, 27),
              random.randint(0, 27))
    x2, y2 = (
        random.randint(0 if x1 < 5 else x1 - 5, 27 if x1 > 22 else x1 + 5),
        random.randint(0 if y1 < 5 else y1 - 5, 27 if y1 > 22 else y1 + 5)
    )
    x_step = 1 if x1 < x2 else -1
    y_step = 1 if y1 < y2 else -1
    for xc in range(x1, x2, x_step):
        for yc in range(y1, y2, y_step):
            color = random.randint(64, 255)
            cv2.circle(image, (xc, yc), radius=0, color=color, thickness=1)


def _add_point_noise(image):
    for _i in range(5):
        x_rand = random.randint(0, 27)
        y_rand = random.randint(0, 27)
        color = random.randint(64, 255)
        cv2.circle(image, (x_rand, y_rand), radius=0, color=color, thickness=1)


def prepare_full_cells(destination_path):
    for i in range(5600):
        _set = 'train' if i < 5000 else 'test'
        path = os.path.join(destination_path, _set, f'{i}.jpg')
        cell = np.zeros((28, 28))
        for row in range(28):
            for col in range(28):
                cell[row][col] = random.randint(0, 255)
        cell = image_util.enhance(cell.astype(np.uint8))
        cv2.imwrite(path, cell)


def prepare_lines_cells(destination_path):
    for i in range(5600):
        _set = 'train' if i < 5000 else 'test'
        path = os.path.join(destination_path, _set, f'{i}.jpg')
        cell = np.zeros((28, 28))
        for j in range(random.randint(1, 10)):
            row1, col1 = random.randint(0, 27), random.randint(0, 27)
            row2, col2 = random.randint(0, 27), random.randint(0, 27)
            cv2.line(cell, (row1, col1), (row2, col2),
                     random.randint(64, 255),
                     thickness=1)
        cell = image_util.enhance(cell.astype(np.uint8))
        cv2.imwrite(path, cell)


def transform_gifs_to_jpgs(path):
    for exam_dir in os.listdir(path):
        exam_dir_path = os.path.join(path, exam_dir)
        print(f'Starting {exam_dir}')
        for file in os.listdir(exam_dir_path):
            try:
                if file.endswith('.gif'):
                    file_path = os.path.join(exam_dir_path, file)
                    image_util.convert_from_gif_to_jpg(file_path,
                                                       file_path[:-4] + '.jpg')
                    os.remove(file_path)
            except Exception:
                print(os.path.join(exam_dir_path, file))
        print(f'Finished {exam_dir}')


def crop_tables(path):
    td = table_detection.TableDetection()
    for exam_dir in os.listdir(path):
        exam_dir_path = os.path.join(path, exam_dir)
        print(f'Starting {exam_dir}')
        if exam_dir.startswith('MI'):
            questions = 25
        else:
            questions = 26
        for file in os.listdir(exam_dir_path):
            try:
                if file.endswith('.jpg'):
                    file_path = os.path.join(exam_dir_path, file)
                    image = cv2.imread(file_path)
                    normalized = normalization.normalize_exam(image)
                    cropped = td.detect_table(normalized, questions)
                    cv2.imwrite(file_path, cropped)
            except Exception:
                print(os.path.join(exam_dir_path, file))
        print(f'Finished {exam_dir}')


def prepare_minus_sign_cells(dataset_root):
    dataset_path_train = os.path.join(dataset_root, 'Minus_sign_cells',
                                      'train')
    dataset_path_test = os.path.join(dataset_root, 'Minus_sign_cells', 'test')
    train_dataset = torchvision.datasets.MNIST(dataset_root, train=True,
                                               download=True)
    train_dataset.data = train_dataset.data[train_dataset.targets == 1]
    train_dataset.targets = train_dataset.targets[train_dataset.targets == 1]

    train_dataset.data = rotate_minus_sign_dataset(data=train_dataset.data)
    transform(train_dataset.data)
    for i in range(len(train_dataset.data)):
        image = train_dataset.data[i].numpy()
        cv2.imwrite(os.path.join(dataset_path_train, f'{i}.jpg'), image)

    test_dataset = torchvision.datasets.MNIST(dataset_root, train=False,
                                              download=True)
    test_dataset.data = test_dataset.data[test_dataset.targets == 1]
    test_dataset.targets = test_dataset.targets[test_dataset.targets == 1]

    test_dataset.data = rotate_minus_sign_dataset(data=test_dataset.data)
    transform(test_dataset.data)
    for i in range(len(test_dataset.data)):
        image = test_dataset.data[i].numpy()
        cv2.imwrite(os.path.join(dataset_path_test, f'{i}.jpg'), image)
