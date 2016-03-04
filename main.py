import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import *
from cs231n.classifiers.cnn2 import *
from cs231n.data_utils import *
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver

parent_dirname = os.path.dirname(os.path.abspath(__file__))
path = parent_dirname + "/datasets/tiny-imagenet-200/"

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

if __name__ == "__main__":

  plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
  plt.rcParams['image.interpolation'] = 'nearest'
  plt.rcParams['image.cmap'] = 'gray'

  # Get tiny imagenet data

  class_names, X_train, y_train, X_val, y_val, X_test, y_test = \
     			load_tiny_imagenet(path);
   
  perm = np.random.permutation(len(X_train))
  X_train = X_train[perm,:]
  y_train = y_train[perm]
  # Overfit Small Data
  num_train = 100000
  data = {
    'X_train': X_train[:num_train],
    'y_train': y_train[:num_train],
    'X_val': X_val,
    'y_val': y_val,
  }
  model = NLayerConvNet(input_dim = (3, 64, 64), num_classes = 200, 
                      weight_scale=0.005, N=2, M=2, num_filters = [64, 64], 
                      filter_sizes = [5, 3], hidden_dims=[200,100], 
                          use_batchnorm= True, reg=0.001)

  solver = Solver(model, data,
                num_epochs=20, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 2e-3,
                },
                verbose=True, print_every=20)
  solver.train()

