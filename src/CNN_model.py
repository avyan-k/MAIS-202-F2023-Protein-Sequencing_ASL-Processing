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
import time
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from itertools import product
from torchinfo import summary
import os
from scipy import stats

DEVICE = ld.load_device()


class MLP_model(nn.Module):

  def __init__(self,layers, neurons_per_layer,dropout=0.5, input_shape = (22,3)):
    super(MLP_model, self).__init__() 
    input_neurons = input_shape[0] * input_shape[1]
    self.dropout = dropout
    self.network = nn.ModuleList()
    self.network.append(nn.Linear(input_neurons, neurons_per_layer))
    for x in range(layers-1):
        self.network.append(nn.Linear(neurons_per_layer, neurons_per_layer*2))
        neurons_per_layer *= 2
    self.network.append(nn.Linear(neurons_per_layer, 27))

  def forward(self, x):
      x = torch.flatten(x, start_dim = 1) # Flatten to a 1D vector
      x = (F.batch_norm(x.T, training=True,running_mean=torch.zeros(x.shape[0]).to(DEVICE),running_var=torch.ones(x.shape[0]).to(DEVICE))).T
      for layer in self.network:
          x = F.leaky_relu(layer(x))
          x = F.dropout(x,self.dropout)
      return x

class CNN_model(nn.Module):

  def __init__(self,numberConvolutionLayers=4,initialKernels=64,numberDense = 0,neuronsDLayer=0,dropout=0.5,channels = 3, classes = 27,image_size = (32,32)):
    
    # Calls the constructor of the parent class (nn.Module) to properly initialize the model
    super(CNN_model, self).__init__() 

    # Convolutional Network
    self.convolutional_network = nn.ModuleList() # list that will store the convolutional layers of the model
    
    
    if numberConvolutionLayers > 0: #only add conv layer if needed
      kernelsPerLayers = initialKernels
        # First convolutional layer
      self.convolutional_network.append(nn.Conv2d(in_channels=channels,out_channels=kernelsPerLayers, kernel_size=3,padding="same")) 
      self.convolutional_network.append(nn.BatchNorm2d(kernelsPerLayers)) # batch normalize the data
      
      for i in range(numberConvolutionLayers - 1):
        # Add convolution layer
        self.convolutional_network.append(nn.Conv2d(in_channels=kernelsPerLayers,out_channels=kernelsPerLayers*2, kernel_size=3,padding="same")) 
        kernelsPerLayers *= 2 # double kernels everytime
        self.convolutional_network.append(nn.BatchNorm2d(kernelsPerLayers)) # batch normalize the data
      # Flattened Output of Convolutional Layers
      flattened = int((image_size[0] / (2**(numberConvolutionLayers))))**2 * kernelsPerLayers 
    else:
      flattened = image_size[0]*image_size[1] * channels # if no conv layers, then flattened is the neurons amount
    
     
    self.dropout_layer = nn.Dropout(p=dropout) # dropout to reduce model and prevent overfitting

    # Dense Network
    self.dense_network = nn.ModuleList() # list that will store the dense layers of the model
    neurons = neuronsDLayer

    self.dense_network.append(nn.Linear(flattened, neurons)) # first dense layer
    self.dense_network.append(nn.BatchNorm1d(neurons)) # batch normalize the data
    
    for i in range(numberDense-1): # one dense layer has already been added
      self.dense_network.append(nn.Linear(neurons, neurons)) # add dense layer
      self.dense_network.append(nn.BatchNorm1d(neurons)) # batch normalize the data

    self.dense_network.append(nn.Linear(neurons, classes)) # classification layer - not counted as part of (hidden)dense network

  def forward(self, x):
    
    '''Forward pass function, needs to be defined for every model'''
    for index,convLayer in enumerate(self.convolutional_network):
      # applies a convolution operation to the input
      x = convLayer(x) 
      # max pool and relu every other conv
      if index % 2 == 1: 
        x = F.relu(x)  
        x = F.max_pool2d(x, 2) 
    x = torch.flatten(x, start_dim = 1) # Flatten to a 1D vector
    x = self.dropout_layer(x) # dropout on some of the convolution

		# fully connected (dense) layers defined in self.dense_network
    for index, dense_layer in enumerate(self.dense_network):
      # dropout on last layer: no ReLU
      if index == len(self.dense_network) - 1: 
        x = self.dropout_layer(x)
        x = dense_layer(x)
      else:
        x = dense_layer(x)
        # relu every other layer
        if index % 2 == 1:
          x = F.relu(x) 

    return x # returns predicted class probabilities for each input
  

def train_model(model,input_shape,train_loader,valid_loader, num_epochs = 200,number_of_validations = 3,learning_rate = 0.001, weight_decay=0.001):
  
  losses = np.empty(num_epochs)
  start = time.time()
  text_file = open(r"results\training\losses.txt", "w",encoding="utf-8")
  text_file.write(f"Attempting {num_epochs} epochs on date of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n with model:")
  model_stats = summary(model, input_shape, verbose=0)
  text_file.write(f"Model Summary:{str(model_stats)}\n")
  text_file.write('\n')
  text_file.close()

  model = model.to(DEVICE)

  # Initializes the Adam optimizer with the model's parameters
  optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
  loss = nn.CrossEntropyLoss().to(DEVICE)
  accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=27).to(DEVICE)
  val_iteration = len(train_loader) // number_of_validations
  for epoch in tqdm(range(num_epochs),desc="Epoch",position=3,leave=False):
  # for epoch in range(num_epochs):
    # Iterate through the training data
    # for iteration, (X_train, y_train) in enumerate(train_loader):
    for iteration, (X_train, y_train) in enumerate(tqdm((train_loader),desc="Iteration",position=4,leave=False)):
      # resets all gradients to 0 after each batch
      optimizer.zero_grad()

      X_train = X_train.to(DEVICE)
      y_train = y_train.to(DEVICE)
      # forward pass of the CNN model on the input data to get predictions
      y_hat = model(X_train)
      
      # comparing the model's predictions with the truth labels
      train_loss = loss(y_hat, y_train)
      
      # backpropagating the loss through the model
      train_loss.backward()

      # takes a step in the direction that minimizes the loss
      optimizer.step()

      # checks if should compute the validation metrics for plotting later
      if iteration % val_iteration == 0 and epoch % 5 == 0:
        valid_model(model,valid_loader,epoch,iteration,accuracy,loss)

    # logging results
    logging_result(train_loss,epoch,start,losses)

  text_file = open(r"results\training\losses.txt", "a") 
  text_file.write(f"Losses: \n{losses}\n")
  text_file.close()
  return losses

def valid_model(cnn,valid_loader,epoch,iteration,accuracy,loss):
  

        
      # stops computing gradients on the validation set
      with torch.no_grad():

        # Keep track of the losses & accuracies
        val_accuracy_sum = 0
        val_loss_sum = 0

        # Make a predictions on the full validation set, batch by batch
        # for X_val, y_val in valid_loader:
        for X_val, y_val in tqdm(valid_loader,desc="Validation Iteration",position=5,leave=False):

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
        # Out to console
        text_file = open(r"results\training\losses.txt", "a") 
        text_file.write(f"\nEPOCH = {epoch} --- ITERATION = {iteration}\n")
        text_file.write(f"Validation loss = {val_loss} --- Validation accuracy = {val_accuracy}\n\n")
        text_file.close()
        # print(f"EPOCH = {epoch} --- ITERATION = {iteration}")
        # print(f"Validation loss = {val_loss} --- Validation accuracy = {val_accuracy}")
        if val_accuracy > 0.96:
          torch.save(cnn.state_dict(), rf"results\training\models\{epoch}-{iteration}-{val_accuracy}.pt")

def logging_result(loss,epoch,start,losses):
  
  training_loss = loss.cpu()
  losses[epoch] = training_loss
  # print(f"\n\nloss: {training_loss.item()} epoch: {epoch}")
  # print("It has now been "+ time.strftime("%Mm%Ss", time.gmtime(time.time() - start))  +"  since the beginning of the program")
  text_file = open(r"results\training\losses.txt", "a")  
  text_file.write(f"loss: {training_loss.item()} epoch: {epoch}\n")
  current =  time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - start))
  text_file.write(f"It has now been {current} since the beginning of the program\n")
  text_file.close()
  
  
def to_see_model(path):
  
  model=torch.load(path, map_location=DEVICE)
  text_file = open(r"our_models/model1.txt", "w") 
  print(model, file=text_file)
  text_file.close()
  
def plot_losses(losses):
  xh = np.arange(0,number_of_epochs)
  plt.plot(xh, losses, color = 'b', marker = ',',label = "Loss") 
  plt.xlabel("Epochs Traversed")
  plt.ylabel("Training Loss")
  plt.grid() 
  plt.legend() 
  plt.show()
  
  
def test(cnn, test_loader):
  
  testing_accuracy_sum = 0
  accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=27).to(DEVICE)
  cnn = cnn.to(DEVICE)
  for (X_test, y_test) in test_loader:
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)
    test_predictions = cnn(X_test)
    testing_accuracy_sum += accuracy(test_predictions, y_test)
  
  test_accuracy = testing_accuracy_sum / len(test_loader)
  
  return test_accuracy


if __name__ == "__main__":
  number_of_epochs = 15         
  # train_loader, valid_loader, test_loader = ld.load_simpleASL_data()
  landmark_train,landmark_validation,land_mark_test = ld.load_landmark_data()
  # test_dict = {}
  # for filename in os.listdir("our_models"):
  #   model_path = os.path.join("our_models", filename)
  #   if os.path.isfile(model_path) and model_path.endswith('.pt'):
  #     cnn = CNN_model(numberConvolutionLayers=4,initialKernels=64,numberDense=0,neuronsDLayer=1024,dropout=0.5).to(DEVICE)
  #     cnn.load_state_dict(torch.load(model_path, map_location = DEVICE))
  #     print(test(cnn, test_loader))
  #     test_dict[model_path] = test(cnn, test_loader)
  # print(max(test_dict, key=test_dict.get))
  # cnn = CNN_model(numberConvolutionLayers=4,initialKernels=64,numberDense=0,neuronsDLayer=1024,dropout=0.5,classes=29,image_size=(128,128)).to(DEVICE)
  mlp = MLP_model(layers = 5, neurons_per_layer = 64,dropout=0, input_shape = (21,2)).to(DEVICE)
  # summary(cnn,(1, 3, 128,128))
  summary(mlp,(1, 2, 21, 1))
  test_dict = {}
  for filename in os.listdir(r"results\training\models"):
    model_path = os.path.join(r"results\training\models", filename)
    if os.path.isfile(model_path) and model_path.endswith('.pt'):
      cnn = mlp = MLP_model(layers = 5, neurons_per_layer = 64,dropout=0, input_shape = (21,2)).to(DEVICE)
      cnn.load_state_dict(torch.load(model_path, map_location = DEVICE))
      print(test(cnn, land_mark_test))
      test_dict[model_path] = test(cnn, land_mark_test)
  if test_dict:
    print(max(test_dict, key=test_dict.get))
  # best_losses = train_model(model=mlp,input_shape=(1, 2, 21, 1),train_loader=landmark_train, valid_loader=landmark_validation,num_epochs=number_of_epochs,number_of_validations = 3,learning_rate=0.001)

  # best_loss = train_model(model=mlp,input_shape=(1, 3, 22, 1),train_loader=landmark_train, valid_loader=landmark_validation,num_epochs=number_of_epochs,num_iterations_before_validation = 2430,learning_rate=0.05)
  # best_losses = train_model(model=cnn,input_shape=(1, 3, 128,128),train_loader=train_loader, valid_loader=valid_loader,num_epochs=number_of_epochs,num_iterations_before_validation = 2430)
  # best_lr, best_nbr_layers, best_neurons_per_layers = 0, 0, 0
  # best_loss = float('-inf')
  # best_losses = []
  # grids = 10
  # for i in tqdm(range (grids),desc="Learning Rate",position=0):
  #       for j in tqdm(range(grids),desc="Layers",position=1,leave=False):
  #             for k in tqdm(range(grids),desc="Neurons",position=2,leave=False):
  #                     lr = np.divide(i+1,100)#such that learning rate is at most 0.1, close to 0
  #                     layers = j
  #                     neurons = (k+1)*10 #neurons from 1 to grids, strictly positive
  #                     mlp = MLP_model(layers = layers, neurons_per_layer = neurons,dropout=0.5, input_shape = (22,3)).to(DEVICE)
  #                     # print(f"learning_rate: {lr} layers: {layers} neurons: {neurons}")
  #                     losses = train_model(model=mlp,input_shape=(1, 3, 22, 1),train_loader=landmark_train, valid_loader=landmark_validation,num_epochs=number_of_epochs,num_iterations_before_validation = 25,learning_rate=lr)
  #                     loss = (stats.trim_mean(losses[10:],0.2))-(stats.trim_mean(losses[:10],0.2))
  #                     if loss > best_loss:
  #                               best_loss = loss
  #                               best_losses = losses
  #                               best_lr = lr
  #                               best_nbr_layers = layers
  #                               best_neurons_per_layers = neurons
  # print(f"Best Parameters: \n learning_rate: {best_lr} layers: {best_nbr_layers} neurons: {best_neurons_per_layers}")
  # to plot the losses
  # xh = np.arange(0,number_of_epochs)
  # plt.plot(xh, best_losses, color = 'b', 
  #        marker = ',',label = "Loss") 
  # plt.xlabel("Epochs Traversed")
  # plt.ylabel("Training Loss")
  # plt.grid() 
  # plt.legend() 
  # plt.show()