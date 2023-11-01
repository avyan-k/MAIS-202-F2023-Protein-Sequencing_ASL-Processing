import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import torchmetrics
import loading_dataset as ld

train_loader, valid_loader, test_loader = ld.load_data()
DEVICE = ld.load_device()
# Define the model

class CNN_model(nn.Module):
  '''
    Class representing a CNN with 2 (convolutional + activation + maxpooling) layers, connected to a single linear layer for prediction
  '''
  def __init__(self,numberConv=3,initialKernels=5,numberDense = 0,neuronsDLayer=0,dropout=0.5):
    super(CNN_model, self).__init__()
    self.convolutional_network = nn.ModuleList()
    kernelsPerLayers = initialKernels
    self.convolutional_network.append(nn.Conv2d(in_channels=3,out_channels=kernelsPerLayers, kernel_size=5,padding="same"))
    for index in range(numberConv-1):
      self.convolutional_network.append(nn.Conv2d(in_channels=kernelsPerLayers,out_channels=kernelsPerLayers*2, kernel_size=5,padding="same"))
      kernelsPerLayers *= 2
    self.flatten = int((512 / (2**numberConv))**2 * initialKernels * 2 **(numberConv-1))
    # self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding="same") # Outputs 5 channels
    # self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding="same") # Outputs 10 channels
    # self.conv3 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding="same") # Outputs 20 channels
    self.linear = nn.Linear(self.flatten, 27) 
    # How did we know that the flattened output will have 490 after 2 convolution layers and 2 maxpool layers? Trial and error! Try running a forward pass with a different number (Not 180)
    # Say you first try 3920: Get an error -> mat1 and mat2 shapes cannot be multiplied (8x180 and 3920x10) -> Now we know each of the 8 samples in the batch has size 180 after flattening
    # We can then change 3920 to 180 :)

  def forward(self, x):
    
    '''Forward pass function, needs to be defined for every model'''
    for convLayer in self.convolutional_network:
      x = convLayer(x)
      x = F.relu(x)
      x = F.max_pool2d(x, 2)
     # 2x2 maxpool


    x = torch.flatten(x, start_dim = 1) # Flatten to a 1D vector
    x = self.linear(x)
    x = F.softmax(x, dim = 1) # dim = 1 to softmax along the rows of the output (We want the probabilities of all classes to sum up to 1)

    return x
  

def train_model(train_loader, valid_loader, test_loader, num_epochs = 2,num_iterations_before_validation = 1000):
  #hyperparameters
  lr_values = {0.01, 0.001}
  cnn_metrics = {}
  cnn_models = {}

  for lr in lr_values:

    cnn_metrics[lr] = {
        "accuracies": [],
        "losses": []
    }

  #loss for multiclass
  loss = nn.CrossEntropyLoss().to(DEVICE)
  #test accuracy, for testing
  accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=27).to(DEVICE) # Regular accuracy


  cnn = CNN_model().to(DEVICE)
  optimizer = optim.Adam(cnn.parameters(), lr)
  cnn_models[lr] = cnn
  
  for epoch in range(num_epochs):

    # Iterate through the training data
    for iteration, (X_train, y_train) in enumerate(train_loader):

      # Move the batch to GPU if it's available
      X_train = X_train.to(DEVICE)
      y_train = y_train.to(DEVICE)

      # The optimizer accumulates the gradient of each weight as we do forward passes -> zero_grad resets all gradients to 0
      optimizer.zero_grad()

      # Compute a forward pass and make a prediction
      y_hat = cnn(X_train)

      # Compute the loss
      train_loss = loss(y_hat, y_train)

      # Compute the gradients in the optimizer
      train_loss.backward()

      # Update the weights
      optimizer.step()

      # Check if should compute the validation metrics for plotting later
      if iteration % num_iterations_before_validation == 0:

        # Don't compute gradients on the validation set
        with torch.no_grad():

          # Keep track of the losses & accuracies
          val_accuracy_sum = 0
          val_loss_sum = 0

          # Make a predictions on the full validation set, batch by batch
          for X_val, y_val in valid_loader:

            # Move the batch to GPU if it's available
            X_val = X_val.to(DEVICE)
            y_val = y_val.to(DEVICE)

            y_hat = cnn(X_val)
            val_accuracy_sum += accuracy(y_hat, y_val)
            val_loss_sum += loss(y_hat, y_val)

          # Divide by the number of iterations (and move back to CPU)
          val_accuracy = (val_accuracy_sum / len(valid_loader)).cpu()
          val_loss = (val_loss_sum / len(valid_loader)).cpu()

          # Store the values in the dictionary
          cnn_metrics[lr]["accuracies"].append(val_accuracy)
          cnn_metrics[lr]["losses"].append(val_loss)

          # Print to console
          print(f"LR = {lr} --- EPOCH = {epoch} --- ITERATION = {iteration}")
          print(f"Validation loss = {val_loss} --- Validation accuracy = {val_accuracy}")
  return cnn_metrics
          
          
def plot_parameter_testing(cnn_metrics,num_iterations_before_validation):
  x_axis = np.arange(0, len(cnn_metrics[0.1]["accuracies"]) * num_iterations_before_validation, num_iterations_before_validation)
  # Plot the accuracies as a function of iterations
  plt.plot(x_axis, cnn_metrics[0.01]["accuracies"], label = "Validation accuracies for lr = 0.01")
  plt.plot(x_axis, cnn_metrics[0.001]["accuracies"], label = "Validation accuracies for lr = 0.001")
  plt.xlabel("Iteration")
  plt.ylabel("Validation accuracy")
  plt.title("Validation accuracy as a function of iteration for CNN")
  plt.legend()

if __name__ == "__main__":
  cnn_metrics = train_model(train_loader, valid_loader, test_loader)
  cnn_metrics, cnn = train_model(train_loader, valid_loader, test_loader)
  plot_parameter_testing(cnn_metrics, 1000)
  MODEL_PATH = r"cnn_model"
  torch.save(cnn.state_dict(), MODEL_PATH)