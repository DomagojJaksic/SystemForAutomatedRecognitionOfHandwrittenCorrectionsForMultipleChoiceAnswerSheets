import os

import cv2
import numpy as np
import torch

from models.custom_classification_model import CustomClassificationModel
from models.densenet_classification_model import DenseNetClassificationModel
from models.random_forest_model import RandomForestModel
from models.resnet_classification_model import ResNetClassificationModel
from models.svm_model import SVMModel
from table_detection import table_detection, table_extraction
from utils import image_util
from utils.image_util import preprocess

from sklearn.metrics import precision_recall_fscore_support

from utils.normalization import normalize_exam

device = torch.device('cuda:0')

empty_labels = [6, 7, 8, 9, 10]


class ExamTester:

    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type
        self.is_cnn = model_type in ['resnet', 'densenet', 'custom']
        self.table_detector = table_detection.TableDetection()
        self.table_extractor = table_extraction.TableExtraction()
        if self.is_cnn:
            self.model = model.to(device)

    def _test_exam(self, path):
        return self._test_exam_image(cv2.imread(path))

    def _test_exam_image(self, image):
        size = 32 if self.model_type == 'densenet' else 28
        cells = self.table_extractor.extract_table(image)
        results = {}
        preprocessed_cells = []
        for i, question in enumerate(cells):
            for j, cell in enumerate(question):
                try:
                    x, y, w, h = cell[0]
                    preprocessed = preprocess(image[y:y + h, x:x + w],
                                              (size, size),
                                              trim_flag=True)
                    preprocessed_cells.append(preprocessed)
                except:
                    dimensions = (size, size)
                    preprocessed_cells.append(-np.ones(dimensions))
        preprocessed_cells = torch.tensor(preprocessed_cells)
        if self.is_cnn:
            preprocessed_cells = preprocessed_cells.div(255.).reshape(
                -1, 1, size, size)
            preprocessed_cells = preprocessed_cells.to(device).float()
            _, predictions = torch.max(self.model(preprocessed_cells), 1)
            predictions = predictions.cpu().numpy()
        else:
            preprocessed_cells = preprocessed_cells.reshape(
                -1, size ** 2) / 255.
            predictions = self.model(preprocessed_cells)

        for i, prediction in enumerate(predictions):
            if torch.max(preprocessed_cells[i]).item() < 0:
                results[(int(i / 3) + 1, i % 3 + 1)] = None
            else:
                results[(int(i / 3) + 1, i % 3 + 1)] = prediction.item()
        return results

    def _test_exams(self, path, labels_path):
        y_true, y_pred = [], []
        y_true_letters, y_pred_letters = [], []
        correct, correct_letters, total, total_letters = 0, 0, 0, 0
        for exam_dir in sorted(os.listdir(path)):
            exam_dir_path = os.path.join(path, exam_dir)
            exam_correct, exam_correct_letters = 0, 0
            exam_total, exam_total_letters = 0, 0
            for file in os.listdir(exam_dir_path):
                try:
                    file_path = os.path.join(exam_dir_path, file)
                    results = self._test_exam(file_path)
                    if labels_path:
                        labels = self.__load_labels(
                            labels_path, exam_dir, f'{file[:-4]}.txt')
                        for key in results:
                            if results[key] is None:
                                continue
                            label, result = labels[key], results[key]

                            if label >= 7:
                                label = 6
                            if result >= 7:
                                result = 6
                            y_true.append(label)
                            y_pred.append(result)

                            if labels[key] != 8:
                                y_true_letters.append(label)
                                y_pred_letters.append(result)
                                total_letters += 1
                                exam_total_letters += 1

                            total += 1
                            exam_total += 1
                            if result == label:
                                correct += 1
                                exam_correct += 1
                                if labels[key] != 8:
                                    correct_letters += 1
                                    exam_correct_letters += 1
                except Exception as e:
                    print(e)
                    print(os.path.join(exam_dir_path, file))

            self.print_exam_statistics(
                exam_total, exam_dir, exam_correct, exam_total_letters,
                exam_correct_letters
            )

        self.print_accuracy_statistics(
            total, correct, total_letters, correct_letters
        )

        precision_macro, recall_macro, f1_macro, _ = (
            precision_recall_fscore_support(y_true, y_pred, average='macro')
        )
        precision_letters_macro, recall_letters_macro, f1_letters_macro, _ = (
            precision_recall_fscore_support(
                y_true_letters, y_pred_letters, average='macro')
        )
        precision_micro, recall_micro, f1_micro, _ = (
            precision_recall_fscore_support(y_true, y_pred, average='micro')
        )
        precision_letters_micro, recall_letters_micro, f1_letters_micro, _ = (
            precision_recall_fscore_support(
                y_true_letters, y_pred_letters, average='micro')
        )

        labels = list(range(0, 7))
        precisions, recalls, f1s, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=labels, zero_division=0)
        precisions_letters, recalls_letters, f1s_letters, _ = (
            precision_recall_fscore_support(
                y_true_letters, y_pred_letters, average=None, labels=labels,
                zero_division=0)
        )

        self.print_precision_recall_f1(
            precision_macro, recall_macro, f1_macro, precision_micro,
            recall_micro, f1_micro
        )
        self.print_precision_recall_f1_letters(
            precision_letters_macro, recall_letters_macro, f1_letters_macro,
            precision_letters_micro, recall_letters_micro, f1_letters_micro
        )
        self.print_precisions_recalls_f1_per_classes(
            precisions, recalls, f1s, precisions_letters, recalls_letters,
            f1s_letters
        )

    def print_exam_statistics(self, exam_total, exam_dir, exam_correct,
                              exam_total_letters, exam_correct_letters):
        if exam_total > 0:
            print(f'{exam_dir} - total accuracy: '
                  f'{round(exam_correct / exam_total, 4)} '
                  f'({exam_correct} / {exam_total})')
        else:
            print(f'{exam_dir} - total accuracy: 0')

        if exam_total_letters > 0:
            print(f'{exam_dir} - letter accuracy: '
                  f'{round(exam_correct_letters / exam_total_letters, 4)} '
                  f'({exam_correct_letters} / {exam_total_letters})')
        else:
            print(f'{exam_dir} - letter accuracy: 0')

    def print_accuracy_statistics(self, total, correct, total_letters,
                                  correct_letters):
        if total > 0:
            print(f'Total accuracy: {round(correct / total, 4)} '
                  f'({correct} / {total})')
        else:
            print(f'Total accuracy: 0')
        if total_letters > 0:
            print(f'Letter accuracy: '
                  f'{round(correct_letters / total_letters, 4)} '
                  f'({correct_letters} / {total_letters})')
        else:
            print(f'Letter accuracy: 0')

    def print_precision_recall_f1(
            self, precision_macro, recall_macro, f1_macro, precision_micro,
            recall_micro, f1_micro):
        print(f'Macro: P = {round(precision_macro, 4)}, '
              f'R = {round(recall_macro, 4)}, '
              f'F1 = {round(f1_macro, 4)}')
        print(f'Micro: P = {round(precision_micro, 4)}, '
              f'R = {round(recall_micro, 4)}, '
              f'F1 = {round(f1_micro, 4)}')

    def print_precision_recall_f1_letters(
            self, precision_letters_macro, recall_letters_macro,
            f1_letters_macro, precision_letters_micro, recall_letters_micro,
            f1_letters_micro):
        print(f'Macro letters: P = {round(precision_letters_macro, 4)}, '
              f'R = {round(recall_letters_macro, 4)}, '
              f'F1 = {round(f1_letters_macro, 4)}')
        print(f'Micro letters: P = {round(precision_letters_micro, 4)},'
              f' R = {round(recall_letters_micro, 4)}, '
              f'F1 = {round(f1_letters_micro, 4)}')

    def print_precisions_recalls_f1_per_classes(
            self, precisions, recalls, f1s, precisions_letters,
            recalls_letters, f1s_letters):
        print(f'Precisions: {list(np.around(precisions, 8))}')
        print(f'Recalls: {list(np.around(recalls, 8))}')
        print(f'F1s: {list(np.around(f1s, 8))}')
        print(f'Precisions letters: {list(np.around(precisions_letters, 8))}')
        print(f'Recalls letters: {list(np.around(recalls_letters, 8))}')
        print(f'F1s letters: {list(np.around(f1s_letters, 8))}')

    def __load_labels(self, path, exam_dir, filename):
        full_path = os.path.join(path, exam_dir, filename)
        labels = {}
        with open(full_path, 'r') as file:
            for line in file:
                x, y, l = line.split(',')
                labels[(int(x), int(y))] = int(l)
        return labels

    def test_exam(self, path, questions):
        image = image_util.convert_from_gif_to_jpg_without_saving(path)
        return self.test_exam_image(image, questions)

    def test_exam_image(self, image, questions):
        size = 32 if self.model_type == 'densenet' else 28
        normalized = normalize_exam(image)
        table = self.table_detector.detect_table(normalized, questions)
        cells = self.table_extractor.extract_table(table)
        preprocessed_cells = []
        for i, question in enumerate(cells):
            for j, cell in enumerate(question):
                try:
                    x, y, w, h = cell[0]
                    preprocessed = preprocess(table[y:y + h, x:x + w],
                                              (size, size), trim_flag=True)
                    preprocessed_cells.append(preprocessed)
                except:
                    pass

        preprocessed_cells = torch.tensor(preprocessed_cells)
        if self.is_cnn:
            preprocessed_cells = preprocessed_cells.div(255.).reshape(
                -1, 1, size, size)
            preprocessed_cells = preprocessed_cells.to(device).float()
            confidences, predictions = torch.max(torch.softmax(
                self.model(preprocessed_cells), 1), 1)
            predictions = predictions.cpu().numpy()
        else:
            preprocessed_cells = preprocessed_cells.reshape(
                -1, size ** 2) / 255.
            confidences = self.model(preprocessed_cells, True)
            predictions = np.argmax(confidences, 1)
            confidences = np.max(confidences, 1)

        return predictions, confidences

    def test_exams(self, path, questions):
        for file in sorted(os.listdir(path)):
            try:
                file_path = os.path.join(path, file)
                predictions, confidences = self.test_exam(file_path, questions)
                for i, prediction in enumerate(predictions):
                    if prediction <= 5:
                        p = chr(65 + prediction)
                    elif confidences[i].item() < 0.5:
                        p = 'No correction'
                    else:
                        continue
                    print("{a}: ({b:2d}, {c}) = {d} ({e:.4f})".format(
                        a=file.split('.')[0], b=int(i / 3) + 1, c=i % 3 + 1,
                        d=p, e=round(confidences[i].item(), 4)
                    ))
            except Exception as e:
                print(e)
                print(os.path.join(path, file))


def test_model(path, model_type):
    if model_type == 'resnet':
        model = ResNetClassificationModel.load_model(path)
    elif model_type == 'custom':
        model = CustomClassificationModel.load_model(path)
    elif model_type == 'densenet':
        model = DenseNetClassificationModel.load_model(path)
    elif model_type == 'svm':
        model = SVMModel.load_model(path)
    elif model_type == 'random_forest':
        model = RandomForestModel.load_model(path)
    else:
        raise ValueError()

    exam_tester = ExamTester(model, model_type)
    exam_tester._test_exams('data/Exams_with_table_cropped/valid/images',
                            'data/Exams_with_table_cropped/valid/labels')
    exam_tester._test_exams('data/Exams_with_table_cropped/test/images',
                            'data/Exams_with_table_cropped/test/labels')
