from abc import abstractmethod, ABC

import pandas as pd
import torch.nn


class NNPattern(ABC, torch.nn.Module):
    """
    Pattern class of the neural network to inherit.

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

        super(NNPattern, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

    @abstractmethod
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
        pass

    @abstractmethod
    def uniformLayer(self, x: torch.Tensor) -> None:
        pass

    @abstractmethod
    def gaussianLayer(self, x: torch.Tensor) -> None:
        pass

    @abstractmethod
    def shuffle_data(self) -> None:
        """
        Update data permutation

        Arguments:
                None

        Outputs:
                None
        """
        pass

    @abstractmethod
    def extract_data(self, df: pd.DataFrame, aim_par: str, split_ratio: float) -> None:
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def train(self, lr: float, num_epochs: int) -> None:
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
        pass

    @abstractmethod
    def predict(self) -> None:
        """
        Validate (test) the neural network

        Arguments:
                None

        Outputs:
                None
        """
        pass

    @abstractmethod
    def train_predict(self) -> None:
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
        pass

    @abstractmethod
    def show_results(self) -> None:
        """
        Demonstrate the loss/accuracy charts

        Arguments:
                None

        Outputs:
                None
        """
        pass
