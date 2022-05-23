from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from abstract_class import NNPattern


class Classification(NNPattern):
    """
    Main class of the neural network to inherit.

    Methods:
            __init__(input_size, output_size)
                Initialize the class

            forward(x)
                forward() function of neural network

            bernoulliLayer()

            extract_data(df, aim_par, split_ratio)
                Prepare given data for training and validating the neural network

            train(lr, num_epochs)
                train() function of neural network

            predict()
                Validate (test) the neural network

            show_results()
                Demonstrate the loss/accuracy charts
    """

    # inputLikely = np.random(0, 1)
    # Applying the bernoulli class
    # data = bernoulli.rvs(size=1000, p=0.8)

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Initialize the class

        Arguments:
                input_size: int
                    The quantity of the input nodes (parameters)
                output_size: int
                    The quantity of the output nodes (parameters)

        Outputs:
                None
        """

        super(Classification, self).__init__(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size

        # self.linear = torch.nn.Linear(self.input_size, self.output_size)
        self.inputLayer = torch.nn.Linear(self.input_size, self.input_size)
        self.uniformLayer(self.inputLayer.weight)
        self.h1 = torch.nn.Tanh()
        self.h2 = torch.nn.Bilinear(self.input_size // 2,
                                    self.input_size - self.input_size // 2, self.input_size)
        self.h3 = torch.nn.Tanh()
        self.outputLayer = torch.nn.Linear(self.input_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward() function of neural network

        Arguments:
                x: torch.Tensor
                    Current state of weights

        Outputs:
                torch.Tensor:
                    Updated weights
        """

        # predict = self.linear(x)
        # return torch.sigmoid(predict)
        x = self.inputLayer(x)
        x = torch.sigmoid(self.h1(x))
        x = torch.relu(self.h1(x))
        x = torch.sigmoid(self.h2(x[:self.input_size // 2], x[self.input_size // 2:]))
        x = torch.relu(self.h3(x))
        x = self.outputLayer(x)
        return torch.sigmoid(x)

    def uniformLayer(self, x: torch.Tensor) -> None:
        torch.nn.init.uniform(x)

    def gaussianLayer(self, x: torch.Tensor) -> None:
        torch.nn.init.normal(x)

    def shuffle_data(self) -> None:
        self.ds_idx = {'train': torch.randperm(self.ds_sizes['train']),
                       'test': torch.randperm(self.ds_sizes['test'])}

    def extract_data(self, data_path: str, aim_par: str, split_ratio: float = .6) -> None:
        """
        Prepare given data for training and validating the neural network

        Arguments:
                df: pd.DataFrame
                    Input data
                aim_par: str
                    Goal parameter (label) that the neural network will predict
                split_ratio: float
                    Estimates the percentage in which train and test datasets will be separated

        Outputs:
                None
        """

        df = pd.read_csv(data_path, delimiter=',')

        self.train_ds = []
        self.test_ds = []
        self.train_target = []
        self.test_target = []
        self.len_ds = df.shape[0]
        self.ds_sizes = {'train': int(self.len_ds * split_ratio),
                         'test': int(self.len_ds * (1. - split_ratio))}

        print(self.ds_sizes)

        df = df.drop(['Unnamed: 0'], axis=1)
        self.targets = {'train': np.array(df[aim_par][:self.ds_sizes['train']]),
                        'test': np.array(df[aim_par][self.ds_sizes['train']:self.len_ds])}
        df = df.drop(aim_par, axis=1)
        self.ds = {'train': np.array(df[:][:self.ds_sizes['train']]),
                   'test': np.array(df[:][self.ds_sizes['train']:self.len_ds])}

        self.ds_idx = {'train': torch.randperm(self.ds_sizes['train']),
                       'test': torch.randperm(self.ds_sizes['test'])}

        self.fig = None

        print(self.ds['train'][0])

    def draw_curve(self, cur_epoch: int, phase_type: int) -> None:
        """
        Create and update loss and accuracy charts of the training phase

        Arguments:
                cur_epoch: int
                    Current epoch of training
                phase_type: int
                    Add train and/or test data to the chart
                    Values:
                        0 - train only
                        1 - train and test

        Outputs:
                None
        """

        sns.set()
        self.x_epoch.append(cur_epoch)
        self.ax0.plot(self.x_epoch, self.y_loss['train'], 'r-', label='train', linewidth=.5)
        self.ax1.plot(self.x_epoch, self.y_err['train'], 'r-', label='train', linewidth=.5)
        if phase_type:
            self.ax0.plot(self.x_epoch, self.y_loss['test'], 'g-', label='test', linewidth=.5)
            self.ax1.plot(self.x_epoch, self.y_err['test'], 'g-', label='test', linewidth=.5)
        if cur_epoch == 0:
            self.ax0.legend()
            self.ax1.grid(False)
            self.ax1.legend()
            self.ax1.grid(False)

    def train(self, lr: float = .0001, num_epochs: int = 50) -> None:
        """
        train() function of neural network

        Arguments:
                lr: float
                    The learning rate of the neural network
                num_epochs: int
                    The quantity of epochs to train the neural network

        Outputs:
                None
        """

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.y_loss = {'train': [], 'test': []}
        self.y_err = {'train': [], 'test': []}
        self.x_epoch = []
        self.fig = plt.figure(figsize=(10, 7), dpi=300)
        self.ax0 = self.fig.add_subplot(121, title="Loss")
        self.ax1 = self.fig.add_subplot(122, title="Accuracy")

        for epoch in range(num_epochs):
            running_loss = 0.
            running_corrects = 0
            for i in self.ds_idx['train']:
                if torch.cuda.is_available():
                    inputs = torch.tensor(self.ds['train'][i],
                                          requires_grad=True, dtype=torch.float).cuda()
                    target = torch.tensor(self.targets['train'][i],
                                          requires_grad=True, dtype=torch.float).cuda()
                else:
                    inputs = torch.tensor(self.ds['train'][i], requires_grad=True, dtype=torch.float)
                    target = torch.tensor(self.targets['train'][i], requires_grad=True, dtype=torch.float)

                optimizer.zero_grad()
                outputs = self.forward(inputs)
                target = target.unsqueeze(0)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_corrects += int(round(outputs.data.item()) == int(target.data.item()))

            epoch_loss = running_loss / self.ds_sizes['train']
            epoch_acc = running_corrects / self.ds_sizes['train']
            self.y_loss['train'].append(epoch_loss)
            self.y_err['train'].append(epoch_acc)
            self.draw_curve(epoch, 0)

            self.shuffle_data()

            print(f'Epoch {epoch + 1}/{num_epochs}:')
            print(f'Loss: {epoch_loss} Acc: {epoch_acc}\n\n')

    def predict(self) -> None:
        """
        Validate (test) the neural network

        Arguments:
                None

        Outputs:
                None
        """

        self.shuffle_data()

        self.test_corrects = 0

        with torch.no_grad():
            for i in self.ds_idx['test']:
                if torch.cuda.is_available():
                    inputs = torch.tensor(self.ds['test'][i], requires_grad=True, dtype=torch.float).cuda()
                    target = torch.tensor(self.targets['test'][i], requires_grad=True, dtype=torch.float).cuda()
                else:
                    inputs = torch.tensor(self.ds['test'][i], requires_grad=True, dtype=torch.float)
                    target = torch.tensor(self.targets['test'][i], requires_grad=True, dtype=torch.float)

                output = self.forward(inputs)
                self.test_corrects += int(round(output.data.item()) == round(target.data.item()))

    def train_predict(self, lr: float = .0001, num_epochs: int = 50) -> None:
        """
        Function for parallel training and validation of the neural network

        Arguments:
             lr: float
                    The learning rate of the neural network
                num_epochs: int
                    The quantity of epochs to train the neural network
        Outputs:
                None
        """

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        self.y_loss = {'train': [], 'test': []}
        self.y_err = {'train': [], 'test': []}
        self.x_epoch = []
        self.fig = plt.figure(figsize=(10, 7), dpi=300)
        self.ax0 = self.fig.add_subplot(121, title="Loss")
        self.ax1 = self.fig.add_subplot(122, title="Accuracy")

        for epoch in range(num_epochs):
            for phase in ['train', 'test']:
                running_loss = 0
                running_corrects = 0

                for i in self.ds_idx[phase]:
                    if torch.cuda.is_available():
                        inputs = torch.tensor(self.ds[phase][i],
                                              requires_grad=True, dtype=torch.float).cuda()
                        target = torch.tensor(self.targets[phase][i],
                                              requires_grad=True, dtype=torch.float).cuda()
                    else:
                        inputs = torch.tensor(self.ds[phase][i], requires_grad=True, dtype=torch.float)
                        target = torch.tensor(self.targets[phase][i], requires_grad=True, dtype=torch.float)

                    optimizer.zero_grad()
                    outputs = self.forward(inputs)
                    target = target.unsqueeze(0)
                    loss = criterion(outputs, target)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()
                    running_corrects += int(round(outputs.data.item()) == int(target.data.item()))

                epoch_loss = running_loss / self.ds_sizes[phase]
                epoch_acc = running_corrects / self.ds_sizes[phase]
                self.y_loss[phase].append(epoch_loss)
                self.y_err[phase].append(epoch_acc)
                if phase == 'test':
                    self.draw_curve(epoch, 1)

                self.shuffle_data()

                print(f'Epoch {epoch + 1}/{num_epochs}:')
                print(f'{phase} Loss: {epoch_loss} Acc: {epoch_acc}\n\n')

        self.predict()

    def show_results(self) -> None:
        """
        Demonstrate the loss/accuracy charts

        Arguments:
                None

        Outputs:
                None
        """

        if self.fig:
            self.fig.show()

        if self.test_corrects:
            colors = sns.color_palette('pastel')[:2]
            plt.pie([self.test_corrects, self.ds_sizes['test'] - self.test_corrects],
                    labels=['Correct', 'False'], colors=colors, autopct='%.2f%%',
                    explode=[0, .2], shadow=True)
            plt.show()
