import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.nn
from torch.autograd import Variable


class LogisticRegr(torch.nn.Module):
    def __init__(self, sizeInTrain, sizeOutTrain):
        super(LogisticRegr, self).__init__()
        self.linear = torch.nn.Linear(sizeInTrain, sizeOutTrain).double()

    def forward(self, train_x):
        predict = self.linear(train_x)
        return torch.sigmoid(predict)
