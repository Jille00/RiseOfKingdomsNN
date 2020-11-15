import torch
import torch.nn.functional as F
import torch.nn as nn

class CNN_Classifier(nn.Module):
    def __init__(self):
        super(CNN_Classifier, self).__init__()

        input_size = 100
        hidden_sizes = [128, 64]
        output_size = 10

        self.main = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                              nn.ReLU(),
                              nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                              nn.ReLU(),
                              nn.Linear(hidden_sizes[1], output_size),
                              nn.LogSoftmax(dim=1))

    def forward(self, x):
        out = self.main(x)
        return out