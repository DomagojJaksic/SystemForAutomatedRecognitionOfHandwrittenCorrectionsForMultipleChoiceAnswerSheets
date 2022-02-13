import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset.dataset
from models import svm_model, random_forest_model
from models.custom_classification_model import CustomClassificationModel
from models.densenet_classification_model import DenseNetClassificationModel
from models.resnet_classification_model import ResNetClassificationModel
from utils import image_util

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(dataset_root, model_name, model_type, merge_train_valid=False):
    size = 32 if model_type == 'densenet' else 28
    x_train, y_train, x_valid, y_valid = dataset.dataset.import_train_dataset(
        dataset_root, merge_train_valid)
    x_test, y_test = dataset.dataset.import_test_dataset(dataset_root)

    if model_type == 'densenet':
        x_train = resize_tensor(x_train)
        x_test = resize_tensor(x_test)
        if not merge_train_valid:
            x_valid = resize_tensor(x_valid)

    train_dataset = dataset.dataset.Dataset(x_train.reshape(-1, 1, size, size),
                                            y_train)
    test_dataset = dataset.dataset.Dataset(x_test.reshape(-1, 1, size, size),
                                           y_test)
    if not merge_train_valid:
        valid_dataset = dataset.dataset.Dataset(
            x_valid.reshape(-1, 1, size, size), y_valid)

    if model_type == 'resnet':
        num_classes = 11
        batch_size = 128
        num_epochs = 10
        lr = 0.01
        lr_step = 1
        momentum = 0.95
        lr_step_gamma = 0.1
        weight_decay = 1e-3
        nesterov = True
        model = ResNetClassificationModel(
            num_classes=num_classes, batch_size=batch_size,
            num_epochs=num_epochs, lr=lr, lr_step=lr_step,
            momentum=momentum, lr_step_gamma=lr_step_gamma,
            weight_decay=weight_decay, nesterov=nesterov
        )
    elif model_type == 'custom':
        num_classes = 11
        batch_size = 128
        num_epochs = 10
        lr = 0.1
        lr_step = 3
        momentum = 0.9
        lr_step_gamma = 0.1
        weight_decay = 1e-5
        nesterov = False
        model = CustomClassificationModel(
            num_classes=num_classes, batch_size=batch_size,
            num_epochs=num_epochs, lr=lr, momentum=momentum, lr_step=lr_step,
            lr_step_gamma=lr_step_gamma, weight_decay=weight_decay
        )
    elif model_type == 'densenet':
        num_classes = 11
        batch_size = 256
        num_epochs = 10
        lr = 0.1
        lr_step = 1
        momentum = 0.9
        lr_step_gamma = 0.1
        weight_decay = 1e-3
        nesterov = True
        model = DenseNetClassificationModel(
            num_classes=num_classes, batch_size=batch_size,
            num_epochs=num_epochs, lr=lr, lr_step=lr_step,
            momentum=momentum, lr_step_gamma=lr_step_gamma,
            weight_decay=weight_decay, nesterov=nesterov
        )
    else:
        raise ValueError('Unsupported model type.')
    model = model.to(device)
    results = model.train(train_dataset,
                          valid_dataset if not merge_train_valid else None)

    with open(f'models/{model_type}_history/{model_name}', 'w+') as file:
        for results_list in results:
            for l in results_list:
                l = l if type(l) == float else l.item()
                file.write(f'{round(l, 4)},')
            file.write('\n')
        file.write(f'num_classes={num_classes}\n')
        file.write(f'batch_size={batch_size}\n')
        file.write(f'num_epochs={num_epochs}\n')
        file.write(f'lr={lr}\n')
        file.write(f'lr_step={lr_step}\n')
        file.write(f'lr_step_gamma={lr_step_gamma}\n')
        file.write(f'momentum={momentum}\n')
        file.write(f'weight_decay={weight_decay}\n')
        file.write(f'nesterov={nesterov}\n')

    if model_type in ['resnet', 'densenet']:
        torch.save(model.model, f'models/{model_type}/{model_name}')
    else:
        torch.save(model, f'models/{model_type}/{model_name}')

    loss, corrects = 0., 0
    dataloader = DataLoader(test_dataset, batch_size=2048, num_workers=4)
    for x, y in dataloader:
        if model_type in ['resnet', 'custom', 'densenet']:
            x = x.to(device)
            y = y.to(device)
        l, c = model.evaluate(x, y)
        loss += l / len(x)
        corrects += c
    print(f'Loss: {loss}')
    print(f'Acc: {corrects / len(x_test)}')
    print(f'Corrects: {corrects}, all_examples: {len(x_test)}')


def fit_model(dataset_root, model_name, model_type, merge_train_valid=False):
    x_train, y_train, x_valid, y_valid = dataset.dataset.import_train_dataset(
        dataset_root, merge_train_valid)
    x_test, y_test = dataset.dataset.import_test_dataset(dataset_root)

    x_train = x_train.reshape(x_train.shape[0],
                              x_train.shape[1] * x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0],
                            x_test.shape[1] * x_test.shape[2])
    if not merge_train_valid:
        x_valid = x_valid.reshape(x_valid.shape[0],
                                  x_valid.shape[1] * x_valid.shape[2])

    if model_type == 'svm':
        kernel = 'rbf'
        c = 100
        gamma = 0.01
        probability = True
        model = svm_model.SVMModel(kernel=kernel, c=c, gamma=gamma,
                                   probability=probability)
    elif model_type == 'random_forest':
        n_estimators = 512
        max_depth = 250
        min_samples_split = 2
        bootstrap = False
        n_jobs = 8
        model = random_forest_model.RandomForestModel(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, bootstrap=bootstrap,
            n_jobs=n_jobs
        )
    else:
        raise ValueError('Unsupported model type.')

    model.fit(x_train.numpy(), y_train.numpy())
    joblib.dump(model.model,
                open(f'models/{model_type}/{model_name}.joblib', 'wb'))

    with open(f'models/{model_type}_history/{model_name}.txt', 'w+') as file:
        if model_type == 'random_forest':
            file.write(f'n_estimators={n_estimators}\n')
            file.write(f'max_depth={max_depth}\n')
            file.write(f'min_samples_split={min_samples_split}\n')
            file.write(f'bootstrap={bootstrap}\n')
            file.write(f'n_jobs={n_jobs}\n')
        elif model_type == 'svm':
            file.write(f'kernel={kernel}\n')
            file.write(f'c={c}\n')
            file.write(f'gamma={gamma}\n')
            file.write(f'probability={probability}\n')

    if not merge_train_valid:
        y_valid_predictions = model.predict(x_valid.numpy())
        correct = np.sum(y_valid_predictions == y_valid.numpy())
        print(correct, len(y_valid))
        print(correct / len(y_valid))

    y_test_predictions = model.predict(x_test.numpy())
    correct = np.sum(y_test_predictions == y_test.numpy())
    print(correct, len(y_test))
    print(correct / len(y_test))


def resize_tensor(tensor):
    tensor_resized = []
    for i in range(len(tensor)):
        image = np.asarray(tensor[i].numpy() * 255, np.uint8)
        image = image_util.resize_image(image, (32, 32)) / 255.
        tensor_resized.append(image)

    tensor = torch.tensor(np.array(tensor_resized)).float()
    return tensor
