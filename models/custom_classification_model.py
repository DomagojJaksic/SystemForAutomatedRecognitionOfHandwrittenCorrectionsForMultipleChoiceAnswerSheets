import copy
import datetime

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomClassificationModel(nn.Module):

    def __init__(self, num_classes=11, batch_size=128, num_epochs=25, lr=0.1,
                 momentum=0.9, lr_step=1, lr_step_gamma=0.1,
                 weight_decay=1e-5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5,
                               stride=1, padding=1, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=5408, out_features=250, bias=True)
        self.fc_logits = nn.Linear(in_features=self.fc1.out_features,
                                   out_features=11, bias=True)

        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.momentum = momentum
        self.lr_step = lr_step
        self.lr_step_gamma = lr_step_gamma
        self.weight_decay = weight_decay

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = SGD(
            [{'params': self.conv1.weight, 'weight_decay': self.weight_decay},
             {'params': self.fc1.weight, 'weight_decay': self.weight_decay},
             {'params': self.fc_logits.weight},
             {'params': self.conv1.bias},
             {'params': self.fc1.bias},
             {'params': self.fc_logits.bias}
             ], lr=self.lr, momentum=self.momentum)
        self.lr_scheduler = StepLR(
            self.optimizer, self.lr_step, self.lr_step_gamma)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        h = self.conv1(x)
        h = torch.relu(h)
        h = self.maxpool1(h)
        h = h.view(h.shape[0], -1)
        h = self.fc1(h)
        h = torch.relu(h)
        return self.fc_logits(h)

    def train(self, train_data, valid_data):
        dataloaders = {
            'train': DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=True, num_workers=1),
        }
        if valid_data is not None:
            dataloaders['val'] = DataLoader(
                valid_data, batch_size=valid_data.data.shape[0], num_workers=1)

        train_acc_history, train_loss_history = [], []
        val_acc_history, val_loss_history = [], []

        best_model_weights = copy.deepcopy(self.state_dict())
        best_acc = 0.
        for epoch in range(self.num_epochs):
            start_time = datetime.datetime.now()
            for phase in ['train', 'val']:
                if phase not in dataloaders:
                    continue
                running_loss = 0.
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs_size = len(inputs)
                    inputs_transformed = inputs.view(inputs_size, 1, 28, 28)
                    inputs_transformed = inputs_transformed.to(device)
                    labels = labels.to(device)
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.forward(inputs_transformed)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    batch_loss, batch_corrects = self.evaluate(
                        inputs_transformed, labels)
                    running_loss += batch_loss
                    running_corrects += batch_corrects

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects / len(
                    dataloaders[phase].dataset)

                print(f'{phase}: Epoch {epoch + 1}, '
                      f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val':
                    val_acc_history.append(epoch_acc)
                    val_loss_history.append(epoch_loss)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_weights = copy.deepcopy(
                            self.state_dict())
                else:
                    train_acc_history.append(epoch_acc)
                    train_loss_history.append(epoch_loss)

            if phase == 'train':
                self.lr_scheduler.step()

            end_time = datetime.datetime.now()
            print(f'Epoch time: {(end_time - start_time).total_seconds()}')

        if valid_data is not None:
            self.load_state_dict(best_model_weights)
        return (train_loss_history, train_acc_history, val_loss_history,
                val_acc_history)

    def evaluate(self, x, y):
        with torch.no_grad():
            criterion = nn.CrossEntropyLoss()
            outputs = self.forward(x)
            loss = criterion(outputs, y)
            _, preds = torch.max(outputs, 1)
            corrects = torch.sum(preds == y)
            return loss.item() * x.size(0), corrects.item()

    def evaluate_image(self, image):
        with torch.no_grad():
            logits = self.forward(image)
            image_class_prediction = (
                (torch.argmax(logits, 1)).detach().numpy()[0]
            )
            confidence = torch.max(torch.softmax(logits, 1))
            return image_class_prediction, confidence

    @staticmethod
    def load_model(path):
        return torch.load(path)
