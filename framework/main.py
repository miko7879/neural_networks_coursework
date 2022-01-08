"""
This contains all code necessary to run a simple neural network using PyTorch on CIFAR - 10 dataset
"""

# Import required libraries
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

"""
This is the main function, it will grab the data and run the network
"""
def main():

    # Grab the data from torchvision and extract the test data
    dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
    test_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())

    # Let's now split the training data into a training and validation dataset - we will start with a validation set of size 5000
    val_size = 5000
    train_size = 45000
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print("Training set size: ", len(train_ds), " Validation set size: ", len(val_ds))

    # Define hyperparameters
    epochs = 50
    b = 128
    lr = 0.01
    mu = 0.9
    print("Epochs: ", epochs, " Batch Size: ", b, " Learning Rate: ", lr, " Mu: ", mu)

    #Create the data loaders to neatly stream our data
    train_loader = DataLoader(train_ds, batch_size = b, shuffle = True, num_workers = 4) #Should shuffle the training data
    val_loader = DataLoader(val_ds, batch_size = b*2, num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size = b*2, num_workers = 4)

    # Instantiate the model
    model = CIFAR_CNN()

    # Check number of parameters
    #print(count_parameters(model))

    # Instantiate the optimizer
    opt = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)

    # Declare the loss
    loss_func = F.cross_entropy

    # Fit the model
    fit(50, model, loss_func, opt, train_loader, val_loader)

    #Calculate loss and accuracy on test set and print
    losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in test_loader])
    test_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
    acc, nums = zip(*[accuracy(model(xb), yb) for xb, yb in test_loader])
    test_acc = np.sum(np.multiply(acc, nums)) / np.sum(nums)
    print("Final results on test set - loss: ", test_loss, " accuracy: ", test_acc)

"""
This will calculate the number of parameters in the network.
"""
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

"""
This will calculate the accuracy
"""
def accuracy(output, y):
    y_hat = torch.argmax(output, dim = 1)
    return (y_hat == y).float().mean(), len(output)

"""
Function to run a backward pass on a batch, returns the loss and the number of items
"""
def loss_batch(model, loss_func, xb, yb, opt = None):
    #Do a forward pass and calculate the loss
    y_hat = model(xb)
    loss = loss_func(y_hat, yb)
    #Run the optimizer
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)

"""
Function to fit the network
"""
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs): #For each epoch
        #Training portion
        model.train()
        for xb, yb in train_dl: #For each batch
            loss_batch(model, loss_func, xb, yb, opt) #Do a forward and backward pass through the network using said batch
        #Eval portion
        model.eval()
        with torch.no_grad(): #Don't compute graidents in this portion
            #Calculate the losses and number of images for each batch, using the zip function to pair up the loss for each batch with the number of outputs
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]) #The '*' operator packs all outputs of loss_batch into the zip
            val_loss = np.sum(np.multiply(losses, nums))/np.sum(nums) #Calculate the validation loss through a weighted average (weight = number of samples in batch)
            print("Epoch: ", epoch, " Validation Loss: ", val_loss)

"""
Define our network
"""
class CIFAR_CNN(nn.Module):

    #Define the network layers in the __init__ method
    def __init__(self):
        super().__init__()

        # --- Part 1 ---
        self.conv1 = nn.Conv2d(3, 9, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(9, 9, kernel_size = 3, stride = 1, padding = 1)

        # --- Part 2 ---
        self.conv3 = nn.Conv2d(9, 18, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(18, 18, kernel_size = 3, stride = 1, padding = 1)

        # --- Part 3 ---
        self.conv5 = nn.Conv2d(18, 36, kernel_size = 3, stride = 1, padding = 1)
        self.conv6 = nn.Conv2d(36, 36, kernel_size = 3, stride = 1, padding = 1)

        # --- Part 4 ---
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(576, 100)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(100, 10)

    #Build up the architecture in the forward pass
    def forward(self, xb):

        # --- Part 1 ---
        xb = F.relu(self.conv1(xb)) #This is the output after the first layer activated with ReLu
        xb = F.relu(self.conv2(xb)) #This is the output after the second layer activated with ReLu
        xb = F.max_pool2d(xb, 2) #Perform 2x2 max pool to reduce dimension from 32x32 to 16x16

        # --- Part 2 ---
        xb = F.relu(self.conv3(xb)) #This is the output after the third layer activated with ReLu
        xb = F.relu(self.conv4(xb)) #This is the output after the fourth layer activated with ReLu
        xb = F.max_pool2d(xb, 2)  #Perform 2x2 max pool to reduce dimension from 16x16 to 8x8

        # --- Part 3 ---
        xb = F.relu(self.conv5(xb))  #This is the output after the fifth layer activated with ReLu
        xb = F.relu(self.conv6(xb))  #This is the output after the sixth layer activated with ReLu
        xb = F.max_pool2d(xb, 2)  #Perform 2x2 max pool to reduce dimension from 8x8 to 4x4

        # --- Part 4 ---
        xb = self.flat(xb) #Flatten to obtain 1x576 dim tensor
        xb = F.relu(self.lin1(xb)) #This is the output after the first dense layer (seventh overall) activated with ReLu
        xb = self.drop(xb) #50% dropout
        return self.out(xb) #This is the output layer

if __name__ == '__main__':
    main()


