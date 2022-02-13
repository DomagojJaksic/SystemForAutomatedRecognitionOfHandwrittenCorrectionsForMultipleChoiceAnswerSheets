from __future__ import division

import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import copy

from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ResNetClassificationModel(nn.Module):

    def __init__(self, num_classes=11, batch_size=16, num_epochs=15, lr=0.01,
                 momentum=0.95, lr_step=1, lr_step_gamma=0.1,
                 weight_decay=1e-5, nesterov=True):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        self.batch_size = batch_size
        self.num_of_epochs = num_epochs
        self.num_classes = num_classes
        self.lr_step = lr_step
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        params_to_update = self.model.parameters()

        self.optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum,
                                   weight_decay=weight_decay,
                                   nesterov=nesterov)
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, lr_step, gamma=lr_step_gamma)
        self.criterion = nn.CrossEntropyLoss()

        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, self.num_classes)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                     bias=False)
        self.model = self.model.to(device)

    def train(self, train_data, valid_data):
        dataloaders = {
            'train': DataLoader(train_data,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=4),
        }
        if valid_data is not None:
            dataloaders['val'] = DataLoader(
                valid_data, batch_size=2048, shuffle=True, num_workers=4)

        train_acc_history, train_loss_history = [], []
        val_acc_history, val_loss_history = [], []

        best_model_weights = copy.deepcopy(self.model.state_dict())
        best_acc = 0.
        print('Start training.')
        for epoch in range(self.num_of_epochs):
            start_time = datetime.datetime.now()
            for phase in ['train', 'val']:

                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                if phase not in dataloaders:
                    continue

                running_loss = 0.
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.forward(inputs)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    batch_loss, batch_corrects = self.evaluate(inputs, labels)
                    running_loss += batch_loss
                    running_corrects += batch_corrects

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(
                    dataloaders[phase].dataset)

                print(f'{phase}: Epoch {epoch + 1}, '
                      f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val':
                    val_acc_history.append(epoch_acc)
                    val_loss_history.append(epoch_loss)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_weights = copy.deepcopy(
                            self.model.state_dict())
                else:
                    train_acc_history.append(epoch_acc)
                    train_loss_history.append(epoch_loss)

            if phase == 'train':
                self.lr_scheduler.step()

            end_time = datetime.datetime.now()
            print(f'Epoch time: {(end_time - start_time).total_seconds()}')

        if valid_data is not None:
            self.model.load_state_dict(best_model_weights)
        return (train_loss_history, train_acc_history, val_loss_history,
                val_acc_history)

    def forward(self, x):
        return self.model(x)

    def evaluate(self, x, y):
        with torch.no_grad():
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            _, preds = torch.max(outputs, 1)
            loss = loss.item() * x.size(0)
            corrects = torch.sum(preds == y.data)
            return loss, corrects

    @staticmethod
    def load_model(path):
        model = ResNetClassificationModel()
        model.model = torch.load(path)
        return model
