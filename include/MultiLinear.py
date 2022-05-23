from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn
from torch.autograd import Variable


class LinearRegression(torch.nn.Module):
    def __init__(self, sizeInTrain, sizeOutTrain):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(sizeInTrain, sizeOutTrain).double()

    def forward(self, train_x):
        return self.linear(train_x)
