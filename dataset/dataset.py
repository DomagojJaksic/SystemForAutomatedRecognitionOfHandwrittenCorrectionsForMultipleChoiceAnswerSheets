import os
import random

import cv2
import numpy as np
import torchvision
from PIL import Image
from torchvision.transforms.functional import rotate, hflip
import torch.utils.data

from utils import image_util


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data=None, targets=None):
        self.data, self.targets = data, targets

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def import_train_dataset(dataset_root, merge_train_valid=False):
    x_train, y_train, x_valid, y_valid = import_correct_letters_dataset_train(
        dataset_root, merge_train_valid)
    x_train2, y_train2, x_valid2, y_valid2 = import_letter_x_train(
        dataset_root, merge_train_valid)
    x_train3, y_train3, x_valid3, y_valid3 = import_minus_sign_train(
        dataset_root, merge_train_valid)
    x_train4, y_train4, x_valid4, y_valid4 = import_empty_cells_train(
        dataset_root, merge_train_valid)
    x_train5, y_train5, x_valid5, y_valid5 = import_full_cells_train(
        dataset_root, merge_train_valid)
    x_train6, y_train6, x_valid6, y_valid6 = import_lines_cells_train(
        dataset_root, merge_train_valid)

    y_train2 = change_label_values(y_train2, 24, 7)
    y_valid2 = change_label_values(y_valid2, 24, 7)
    y_train3 = change_label_values(y_train3, 1, 8)
    y_valid3 = change_label_values(y_valid3, 1, 8)

    x_train = torch.cat(
        (x_train, x_train2, x_train3, x_train4, x_train5, x_train6), 0)
    x_valid = torch.cat(
        (x_valid, x_valid2, x_valid3, x_valid4, x_valid5, x_valid6), 0)
    y_train = torch.cat(
        (y_train, y_train2, y_train3, y_train4, y_train5, y_train6), 0)
    y_valid = torch.cat(
        (y_valid, y_valid2, y_valid3, y_valid4, y_valid5, y_valid6), 0)

    return x_train, y_train - 1, x_valid, y_valid - 1


def import_correct_letters_dataset_train(dataset_root,
                                         merge_train_valid=False):
    train_dataset = torchvision.datasets.EMNIST(dataset_root, train=True,
                                                download=True, split='letters')

    train_dataset.data = train_dataset.data[train_dataset.targets <= 6]
    train_dataset.targets = train_dataset.targets[train_dataset.targets <= 6]
    return get_train_and_valid_data_from_dataset(
        train_dataset.data, train_dataset.targets,
        split=1 if merge_train_valid else 0.9
    )


def import_letter_x_train(dataset_root, merge_train_valid=False):
    train_dataset = torchvision.datasets.EMNIST(dataset_root, train=True,
                                                download=True, split='letters')

    train_dataset.data = train_dataset.data[train_dataset.targets == 24]
    train_dataset.targets = train_dataset.targets[train_dataset.targets == 24]
    return get_train_and_valid_data_from_dataset(
        train_dataset.data, train_dataset.targets,
        split=1 if merge_train_valid else 0.9
    )


def import_minus_sign_train(dataset_root, merge_train_valid=False):
    dataset_path = os.path.join(
        dataset_root, 'Minus_sign_cells', 'train')
    data = []
    for file in os.listdir(dataset_path):
        try:
            file_path = os.path.join(dataset_path, file)
            data.append(image_util.grayscale(cv2.imread(file_path)))
        except Exception as e:
            print(e)
            print(os.path.join(dataset_path, file))

    data = torch.tensor(np.array(data))
    targets = torch.tensor([1 for i in range(len(data))])
    return get_train_and_valid_data_from_dataset(
        data, targets, False, split=1 if merge_train_valid else 0.9
    )


def import_empty_cells_train(dataset_root, merge_train_valid=False):
    dataset_root = os.path.join(
        dataset_root, 'Empty_cells', 'train', 'processed')
    data = []
    for exam_dir in os.listdir(dataset_root):
        exam_dir_path = os.path.join(dataset_root, exam_dir)
        for file in os.listdir(exam_dir_path):
            try:
                file_path = os.path.join(exam_dir_path, file)
                data.append(image_util.grayscale(cv2.imread(file_path)))
            except Exception as e:
                print(e)
                print(os.path.join(exam_dir_path, file))

    data = torch.tensor(np.array(data))
    targets = torch.tensor([9 for i in range(len(data))])
    return get_train_and_valid_data_from_dataset(
        data, targets, False, split=1 if merge_train_valid else 0.9
    )


def import_full_cells_train(dataset_root, merge_train_valid=False):
    path = os.path.join(dataset_root, 'Invalid_cells/full', 'train')
    data = []
    for file in os.listdir(path):
        try:
            file_path = os.path.join(path, file)
            data.append(image_util.grayscale(cv2.imread(file_path)))
        except Exception as e:
            print(e)
            print(os.path.join(path, file))
    data = torch.tensor(np.array(data))
    targets = torch.tensor([10 for i in range(len(data))])
    return get_train_and_valid_data_from_dataset(
        data, targets, False, split=1 if merge_train_valid else 0.9
    )


def import_lines_cells_train(dataset_root, merge_train_valid=False):
    path = os.path.join(dataset_root, 'Invalid_cells/lines', 'train')
    data = []
    for file in os.listdir(path):
        try:
            file_path = os.path.join(path, file)
            data.append(image_util.grayscale(cv2.imread(file_path)))
        except Exception as e:
            print(e)
            print(os.path.join(path, file))
    data = torch.tensor(np.array(data))
    targets = torch.tensor([11 for i in range(len(data))])
    return get_train_and_valid_data_from_dataset(
        data, targets, False, split=1 if merge_train_valid else 0.9
    )


def get_train_and_valid_data_from_dataset(data, targets, transformation=True,
                                          split=0.9):
    N = int(split * data.shape[0])

    x_train, y_train = data[:N], targets[:N]
    x_valid, y_valid = data[N:], targets[N:]
    x_train, x_valid = x_train.float().div_(255.0), x_valid.float().div_(255.0)

    if transformation:
        transform(x_train)
        transform(x_valid)

    return x_train, y_train, x_valid, y_valid


def import_test_dataset(dataset_root):
    x_test, y_test = import_correct_letters_dataset_test(dataset_root)
    x_test2, y_test2 = import_letter_x_test(dataset_root)
    x_test3, y_test3 = import_minus_sign_test(dataset_root)
    x_test4, y_test4 = import_empty_cells_test(dataset_root)
    x_test5, y_test5 = import_full_cells_test(dataset_root)
    x_test6, y_test6 = import_lines_cells_test(dataset_root)

    y_test2 = change_label_values(y_test2, 24, 7)
    y_test3 = change_label_values(y_test3, 1, 8)

    x_test = torch.cat(
        (x_test, x_test2, x_test3, x_test4, x_test5, x_test6), 0)
    y_test = torch.cat(
        (y_test, y_test2, y_test3, y_test4, y_test5, y_test6), 0)

    return x_test, y_test - 1


def import_correct_letters_dataset_test(dataset_root):
    test_dataset = torchvision.datasets.EMNIST(dataset_root, train=False,
                                               download=True, split='letters')

    test_dataset.data = test_dataset.data[test_dataset.targets <= 6]
    test_dataset.targets = test_dataset.targets[test_dataset.targets <= 6]
    return get_test_data_from_dataset(test_dataset.data, test_dataset.targets)


def import_letter_x_test(dataset_root):
    test_dataset = torchvision.datasets.EMNIST(dataset_root, train=False,
                                               download=True, split='letters')

    test_dataset.data = test_dataset.data[test_dataset.targets == 24]
    test_dataset.targets = test_dataset.targets[test_dataset.targets == 24]
    return get_test_data_from_dataset(test_dataset.data, test_dataset.targets)


def import_minus_sign_test(dataset_root):
    dataset_path = os.path.join(
        dataset_root, 'Minus_sign_cells', 'test')
    data = []
    for file in os.listdir(dataset_path):
        try:
            file_path = os.path.join(dataset_path, file)
            data.append(image_util.grayscale(cv2.imread(file_path)))
        except Exception as e:
            print(e)
            print(os.path.join(dataset_path, file))

    data = torch.tensor(np.array(data))
    targets = torch.tensor([1 for i in range(len(data))])
    return get_test_data_from_dataset(data, targets, False)


def import_empty_cells_test(dataset_root):
    dataset_root = os.path.join(
        dataset_root, 'Empty_cells', 'test', 'processed')
    data = []
    for exam_dir in os.listdir(dataset_root):
        exam_dir_path = os.path.join(dataset_root, exam_dir)
        for file in os.listdir(exam_dir_path):
            try:
                file_path = os.path.join(exam_dir_path, file)
                data.append(image_util.grayscale(cv2.imread(file_path)))
            except Exception as e:
                print(e)
                print(os.path.join(exam_dir_path, file))

    data = torch.tensor(np.array(data))
    targets = torch.tensor([9 for i in range(len(data))])
    return get_test_data_from_dataset(data, targets, False)


def import_full_cells_test(dataset_root):
    path = os.path.join(dataset_root, 'Invalid_cells/full', 'test')
    data = []
    for file in os.listdir(path):
        try:
            file_path = os.path.join(path, file)
            data.append(image_util.grayscale(cv2.imread(file_path)))
        except Exception as e:
            print(e)
            print(os.path.join(path, file))
    data = torch.tensor(np.array(data))
    targets = torch.tensor([10 for i in range(len(data))])
    return get_test_data_from_dataset(data, targets, False)


def import_lines_cells_test(dataset_root):
    path = os.path.join(dataset_root, 'Invalid_cells/lines', 'test')
    data = []
    for file in os.listdir(path):
        try:
            file_path = os.path.join(path, file)
            data.append(image_util.grayscale(cv2.imread(file_path)))
        except Exception as e:
            print(e)
            print(os.path.join(path, file))
    data = torch.tensor(np.array(data))
    targets = torch.tensor([11 for i in range(len(data))])
    return get_test_data_from_dataset(data, targets, False)


def get_test_data_from_dataset(data, targets, transformation=True):
    x_test, y_test = data, targets
    x_test = x_test.float().div_(255.0)
    if transformation:
        transform(x_test)
    return x_test, y_test


def transform(data):
    for i in range(data.shape[0]):
        x = data[i]
        x = hflip(x)
        img = Image.fromarray(x.numpy())
        img = rotate(img, 90)
        x = torch.from_numpy(np.array(img))
        data[i] = x


def change_label_values(labels, old, new):
    return torch.where(labels == old, new, old)


def rotate_minus_sign_dataset(data):
    for i in range(len(data)):
        image = data[i].numpy()
        if random.uniform(0, 1) > 0.5:
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(
                image_center, random.randint(-45, 45), 1.0)
            image = cv2.warpAffine(image, rot_mat, image.shape[1::-1],
                                   flags=cv2.INTER_LINEAR)
        if random.uniform(0, 1) > 0.5:
            image = image_util.convert(image_util.resize_image(
                image_util.trim(image_util.convert(image))))
        data[i] = torch.tensor(image)
    return data
