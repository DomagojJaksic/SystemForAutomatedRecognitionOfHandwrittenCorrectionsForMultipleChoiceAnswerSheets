import sys

from models.custom_classification_model import CustomClassificationModel
from models.densenet_classification_model import DenseNetClassificationModel
from models.random_forest_model import RandomForestModel
from models.resnet_classification_model import ResNetClassificationModel
from models.svm_model import SVMModel

from utils import test_utils

if __name__ == '__main__':
    try:
        path = sys.argv[1]
        questions = int(sys.argv[2])
        model_type = sys.argv[3]
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
    except ValueError as e:
        print('Wrong parameters.')
        exit(1)
    except Exception as e:
        print('Error occurred.')
        exit(1)
