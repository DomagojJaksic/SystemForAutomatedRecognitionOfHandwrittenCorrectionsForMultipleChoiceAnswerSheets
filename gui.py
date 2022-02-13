import datetime
from tkinter import *
from tkinter.scrolledtext import ScrolledText

import sys

from io import StringIO

from models.custom_classification_model import CustomClassificationModel
from models.densenet_classification_model import DenseNetClassificationModel
from models.random_forest_model import RandomForestModel
from models.resnet_classification_model import ResNetClassificationModel
from models.svm_model import SVMModel
from utils import test_utils


class Application(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(anchor=N, fill=BOTH, expand=False, padx=100, pady=3)

        self.init_dataset_path_widgets()
        self.init_model_widgets(master)
        self.init_questions_widgets()
        self.init_start_button()
        self.master.geometry(str(400) + "x" + str(325))

    def callback(self, P):
        if str.isdigit(P) or P == "":
            return True
        else:
            return False

    def init_dataset_path_widgets(self):
        self.dataset_path_label_text = StringVar()
        self.dataset_path_label_text.set("Enter the exams path:")
        self.dataset_path_label = Label(
            textvariable=self.dataset_path_label_text)

        self.dataset_path_entry_text = StringVar()
        self.dataset_path_entry = Entry(
            width=50, textvariable=self.dataset_path_entry_text)
        self.dataset_path_entry_text.set("")

        self.dataset_path_error_text = StringVar()
        self.dataset_path_error_text.set("")
        self.dataset_path_error = Label(
            textvariable=self.dataset_path_error_text)

        self.dataset_path_label.pack(side=TOP, anchor=W, padx=5)
        self.dataset_path_entry.pack(side=TOP, anchor=W, padx=5)
        self.dataset_path_error.pack(side=TOP, anchor=W, padx=5)

    def init_model_widgets(self, master):
        self.choose_model_label_text = StringVar()
        self.choose_model_label_text.set("Choose a model:")
        self.choose_model_label = Label(
            textvariable=self.choose_model_label_text)
        self.choose_model_label.pack(side=TOP, anchor=W, padx=5)

        self.model_type = StringVar()
        Radiobutton(master, text="ResNet model", variable=self.model_type,
                    value='resnet').pack(side=TOP, anchor=W)
        Radiobutton(master, text="DenseNet model", variable=self.model_type,
                    value='densenet').pack(side=TOP, anchor=W)
        Radiobutton(master, text="Custom CNN model", variable=self.model_type,
                    value='custom').pack(side=TOP, anchor=W)
        Radiobutton(master, text="SVM model", variable=self.model_type,
                    value='svm').pack(side=TOP, anchor=W)
        Radiobutton(master, text="Random forest model",
                    variable=self.model_type, value='random_forest').pack(
            side=TOP, anchor=W, pady=(0, 5))

    def init_questions_widgets(self):
        self.questions_label_text = StringVar()
        self.questions_label_text.set("Enter the number of questions:")
        self.questions_label = Label(
            textvariable=self.questions_label_text)

        vcmd = (self.register(self.callback))

        self.questions_entry_text = StringVar()
        self.questions_entry = Entry(
            width=50, textvariable=self.questions_entry_text, validate='all',
            validatecommand=(vcmd, '%P'))
        self.questions_entry_text.set("")

        self.questions_label.pack(side=TOP, anchor=W, padx=5)
        self.questions_entry.pack(side=TOP, anchor=W, padx=5, pady=(0, 5))

    def init_start_button(self):
        self.start_button = Button(
            text="Start",
            command=lambda: self.start_evaluation(
                self.dataset_path_entry_text.get(),
                int(self.questions_entry.get()),
                self.model_type.get()
            )
        )
        self.start_button.pack(side=TOP, anchor=CENTER, padx=5, pady=10)

    def start_evaluation(self, path, questions, model_type):
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        self.start_button.config(state='disabled')
        self.start_button.update()
        self.master.geometry(str(400) + "x" + str(835))
        self.master.update()

        try:
            self.results_text_box.pack_forget()
            self.results_text_box.update()
        except:
            pass
        self.results_text_box = ScrolledText(master=self.master, height=30)
        self.results_text_box.insert(INSERT, "Processing data...\n")
        self.results_text_box.pack(anchor=N, fill=BOTH, expand=False)
        self.results_text_box.update()

        print('Start.')
        self.evaluate_exams(path, questions, model_type)
        print('Done.')
        sys.stdout = old_stdout
        mystdout.seek(0)
        output = mystdout.read()
        self.results_text_box.delete('1.0', END)
        self.results_text_box.insert(INSERT, output)
        self.start_button.config(state='normal')
        self.start_button.update()

    def evaluate_exams(self, path, questions, model_type):
        start_time = datetime.datetime.now()
        try:
            if model_type == 'resnet':
                model = ResNetClassificationModel.load_model(
                    'models/resnet/resnet_model_full.txt')
            elif model_type == 'densenet':
                model = DenseNetClassificationModel.load_model(
                    'models/densenet/densenet_model_full.txt')
            elif model_type == 'custom':
                model = CustomClassificationModel.load_model(
                    'models/custom/custom_model_full.txt')
            elif model_type == 'random_forest':
                model = RandomForestModel.load_model(
                    'models/random_forest/random_forest_model_full.joblib')
            elif model_type == 'svm':
                model = SVMModel.load_model('models/svm/svm_model_full.joblib')
            else:
                raise ValueError()
            exam_tester = test_utils.ExamTester(model, model_type=model_type)
            exam_tester.test_exams(path, questions)
        except ValueError:
            print('Wrong parameters.')
            exit(1)
        except Exception:
            print('Error occurred.')
            exit(1)
        print((datetime.datetime.now() - start_time).total_seconds())


root = Tk()
root.title('Exam evaluator')
app = Application(master=root)
app.mainloop()
