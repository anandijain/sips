import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, in_dim, out_dim, mid_dim=50, classify=True):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, mid_dim)
        self.fc4 = nn.Linear(mid_dim, mid_dim)
        self.fc5 = nn.Linear(mid_dim, mid_dim)
        self.fc6 = nn.Linear(mid_dim, out_dim)

        self.classify = classify
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc4(x)
        x = self.fc5(x)
        if self.classify:
            return self.softmax(self.fc6(x))
        else:
            return F.relu(self.fc6(x))
