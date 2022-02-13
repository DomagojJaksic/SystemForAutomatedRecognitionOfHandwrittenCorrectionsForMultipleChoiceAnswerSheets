import os
from utils.image_util import convert_from_gif_to_jpg


def convert_exams(path):
    new_path = path + '_jpg'
    create_dir_if_needed(new_path)
    for exam_dir in os.listdir(path):
        exam_dir_path = os.path.join(path, exam_dir)
        new_exam_dir_path = os.path.join(new_path, exam_dir)
        create_dir_if_needed(new_exam_dir_path)
        for file in os.listdir(exam_dir_path):
            file_path = os.path.join(exam_dir_path, file)
            new_file_path = os.path.join(new_exam_dir_path,
                                         str(file.split('.')[0]) + '.jpg')
            convert_exam(file_path, new_file_path)


def create_dir_if_needed(path):
    if not os.path.exists(path):
        os.mkdir(path)


def convert_exam(path, new_path):
    convert_from_gif_to_jpg(path, new_path)
