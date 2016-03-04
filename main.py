import os
import sys
import numpy as np
from data_utils import *

parent_dirname = os.path.dirname(os.path.abspath(__file__))
path = parent_dirname + "/datasets/tiny-imagenet-200/"

if __name__ == "__main__":

  class_names, X_train, y_train, X_val, y_val, X_test, y_test = \
     			load_tiny_imagenet(path);
  print X_train.size
  print y_train.size
  print X_val.size
  print y_val.size
