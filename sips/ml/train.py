import time

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RUNNING_INTERVAL = 10


class Model(nn.Module):
    def __init__(self, in_dim, out_dim, mid_dim=20, classify=True):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, mid_dim)
        # self.fc4 = nn.Linear(mid_dim, mid_dim)
        self.fc6 = nn.Linear(mid_dim, out_dim)

        self.classify = classify
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.fc4(x)
        if self.classify:
            return self.softmax(self.fc6(x))
        else:
            return F.relu(self.fc6(x))


def train(d, model_name, epochs=20):
    for epoch in range(epochs):
        train_epoch(d, epoch=epoch, verbose=False)
        test_epoch(d, epoch=epoch, verbose=False)
        
        torch.save(d["model"].state_dict(), model_name + '.pth')


def train_epoch(d, epoch, verbose=False):
    # requires model, train_loader, optimizer, criterion, writer, classify
    print(f"training: {epoch}")
    d["model"].train()
    correct = 0
    total = 0
    running_loss = 0.0
    for i, data in enumerate(d["train_loader"], 0):
        x, y = data["x"].to(device), data["y"].to(device)

        # model stuff
        d["optimizer"].zero_grad()
        y_hat = d["model"](x)
        if verbose:
            print(f"y_hat: {y_hat}")
            print(f"y: {y}")

        if d["classify"]:
            class_idxs = torch.max(y, 1)[1]
            loss = d["criterion"]((y_hat), class_idxs)
        else:
            loss = d["criterion"]((y_hat), y)

        loss.backward()
        d["optimizer"].step()
        if verbose:
            print(f"loss: {loss}")

        d["writer"].add_scalar("train_loss", loss, i +
                               epoch * len(d["train_loader"]))
        running_loss += loss.item()

        # accuracy
        if d["classify"]:
            preds = torch.max(y_hat, 1)[1]
            batch_size = y.size(0)
            total += batch_size

            batch_correct = (preds == class_idxs).sum().item()
            correct += batch_correct
            d["writer"].add_scalar(
                "train_acc",
                batch_correct / batch_size,
                i + epoch * len(d["train_loader"]),
            )

        if i % RUNNING_INTERVAL == RUNNING_INTERVAL - 1:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}")
            print(f"y: {y}, y_hat: {y_hat}")
            running_loss = 0.0

    if d["classify"]:
        print(f"train accuracy {(100 * correct / total):.2f} %")


def test_epoch(d, epoch, verbose=False):
    """ 
    requires model, test_loader, criterion, writer
    and requires dataloader to return {'x', 'y'} dict

    """
    print("testing")
    d["model"].eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for i, test_data, in enumerate(d["test_loader"], 0):
            test_x, test_y = test_data["x"].to(
                device), test_data["y"].to(device)

            test_y_hat = d["model"](test_x)
            if verbose:
                print(f"test_y_hat: {test_y_hat}")
                print(f"test_y: {test_y}")



            if d["classify"]:
                class_idxs = torch.max(test_y, 1)[1]
                test_loss = d["criterion"]((test_y_hat), class_idxs)
            else:
                test_loss = d["criterion"]((test_y_hat), test_y)

            d["writer"].add_scalar(
                "test_loss", test_loss, i + epoch * len(d["test_loader"])
            )

            if d['classify']:
                batch_size = test_y.size(0)
                total += batch_size
                _, predicted = torch.max(test_y_hat.data, 1)
                test_loss = d["criterion"](test_y_hat, class_idxs)

                batch_correct = (predicted == class_idxs).sum().item()
                correct += batch_correct
                d["writer"].add_scalar(
                    "test_acc",
                    batch_correct / batch_size,
                    i + epoch * len(d["test_loader"]),
                )

            running_loss += test_loss.item()
            if i % RUNNING_INTERVAL == RUNNING_INTERVAL - 1:
                # print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}")
                print(f"test_y: {test_y}, test_y_hat: {test_y_hat}")
                running_loss = 0.0
    if d['classify']:
        print(f"test accuracy {(100 * correct / total):.2f} %")


def prep_loader(trainset, testset, model_name, batch_size=1, classify=False):

    x, y = trainset[0].values()

    train_loader = DataLoader(trainset, batch_size=batch_size)
    test_loader = DataLoader(testset, batch_size=batch_size)

    writer = SummaryWriter(f"runs/{model_name}{time.asctime()}")

    model = Model(in_dim=x.shape[0], out_dim=y.shape[0],
                  classify=classify).to(device)

    if classify:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters())  # , lr=0.0001)
    d = {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "criterion": criterion,
        "optimizer": optimizer,
        "model": model,
        "writer": writer,
        "classify": classify,
    }
    return d
