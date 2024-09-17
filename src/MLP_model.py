import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import torchmetrics
import loading_dataset as ld
import time
from datetime import datetime
from torchinfo import summary
import os

DEVICE = ld.load_device()



class MLP_model(nn.Module):
  """Our MLP model
  Args:
      layers: total num of layers (excluding the output layer)
      neurons_per_layer: num of neurons per layer
      dropout: % of neurons dropping our after each layer
      input_shape: shape of the input array [21,2] (21 hand landmarks and 2 dimensions (x,y))
      network: Module list of layers
  """
  def __init__(self,layers, neurons_per_layer,dropout=0.5, input_shape = (21,2)):
    
    super(MLP_model, self).__init__() 
    input_neurons = input_shape[0] * input_shape[1] # total num of neurons in input layer
    self.dropout = dropout
    self.network = nn.ModuleList() # initialize empty module list
    self.network.append(nn.Linear(input_neurons, neurons_per_layer)) # input layer
    
    for x in range(layers-1): # hidden layers 
        self.network.append(nn.Linear(neurons_per_layer, neurons_per_layer*2))
        neurons_per_layer *= 2
    self.network.append(nn.Linear(neurons_per_layer, 27)) # output layers with 27 classes
  
  def forward(self, x):
      x = torch.flatten(x, start_dim = 1) # Flatten to a 1D vector
      x = (F.batch_norm(x.T, training=True,running_mean=torch.zeros(x.shape[0]).to(DEVICE),running_var=torch.ones(x.shape[0]).to(DEVICE))).T
      
      for layer in self.network:
          x = F.leaky_relu(layer(x)) # ReLU activation function
          x = F.dropout(x,self.dropout)
      return x
    
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
  
def plot_losses(losses):
  xh = np.arange(0,number_of_epochs)
  plt.plot(xh, losses, color = 'b', marker = ',',label = "Loss") 
  plt.xlabel("Epochs Traversed")
  plt.ylabel("Training Loss")
  plt.grid() 
  plt.legend() 
  plt.show()
  
# Get accuracy
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
  landmark_train,landmark_validation,land_mark_test = ld.load_landmark_data()
  mlp = MLP_model(layers = 5, neurons_per_layer = 64,dropout=0, input_shape = (21,2)).to(DEVICE)
  summary(mlp,(1, 2, 21, 1))
  mlp.load_state_dict(torch.load(r'our_models\MLP\model3.pt', map_location = DEVICE))
  print(test(mlp, land_mark_test))
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
