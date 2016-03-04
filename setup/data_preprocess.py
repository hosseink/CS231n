import sys
import os
import json
import codecs
import pickle
import glob
import numpy as np

parent_dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = parent_dirname + "/datasets/tiny-imagenet-200/"

def get_idx_to_label(dirname):
  ids = []
  for filename in os.listdir(dirname):
    ids.append(filename)
  return {k:v for v,k in enumerate(ids)}
  

def create_train_table(dirname, idx_to_label):
  table = []
  for filename in os.listdir(dirname):
    label = idx_to_label[filename] 
    class_dir = dirname + filename + '/'
    images_dir = class_dir + 'images/'
    bb_path = class_dir+ filename + '_boxes.txt'
    with open(bb_path,'r') as f:
        for line in f:
            line =  line.strip().split('\t')
            image_path = images_dir + line[0]
            table.append([image_path, str(label), 
                          line[1], line[2], line[3], line[4]]) 
  print len(table)
  perm = np.random.permutation(len(table))
  table = [table[i] for i in perm]
  return table
  
def create_val_table(dirname, idx_to_label):
  table = []
  images_dir = dirname + 'images/'
  bb_path = dirname+ 'val_annotations.txt'
  with open(bb_path,'r') as f:
    for line in f:
      line =  line.strip().split('\t')
      image_path = images_dir + line[0]
      table.append([image_path, str(idx_to_label[line[1]]), 
      		  line[2], line[3], line[4], line[5]]) 
  print len(table)
  return table
  
def create_test_table(dirname):
  table = []
  images_dir = dirname + 'images/'
  for image_name in os.listdir(images_dir):
    table.append([images_dir+image_name, image_name])
  print len(table)
  return table

def create_label_table(filename, idx_to_label):
  table = []
  with open(filename, 'r') as f:
    for line in f:
      line = line.strip().split('\t')
      if not line[0] in idx_to_label:
        continue
      table.append([str(idx_to_label[line[0]]), line[0], line[1]])
  print len(table)
  return table;


if __name__ == "__main__":

  # Creating tables
  idx_to_label = get_idx_to_label(dataset_dir + "train/")
  train_table = create_train_table(dataset_dir+'train/', idx_to_label)
  val_table = create_val_table(dataset_dir+'val/', idx_to_label)
  test_table = create_test_table(dataset_dir+'test/')
  label_table = create_label_table(dataset_dir+'words.txt', idx_to_label)

  table_dir = parent_dirname + '/datasets/' + 'tables/'
  if not os.path.isdir(table_dir):
    os.system('mkdir '+ table_dir)

  train_table_file = open(table_dir + 'train_table.txt', 'w')
  train_table_file.write('\n'.join(['\t'.join(t) for t in train_table]))
  
  val_table_file = open(table_dir + 'val_table.txt', 'w')
  val_table_file.write('\n'.join(['\t'.join(t) for t in val_table]))

  test_table_file = open(table_dir + 'test_table.txt', 'w')
  test_table_file.write('\n'.join(['\t'.join(t) for t in test_table]))
  
  words_file = open(table_dir + 'words.txt', 'w')
  words_file.write('\n'.join(['\t'.join(t) for t in label_table]))
