#!/bin/bash

#Removing any data directory

datasets=datasets
rm -rf $datasets
mkdir $datasets

# Get tiny Imagenet data
echo "==========================================="
echo "=       Getting tiny Imagenet dataset     ="
echo "==========================================="
tinyImageNet=tiny-imagenet-200.zip
curl -O http://cs231n.stanford.edu/$tinyImageNet
unzip -d $datasets $tinyImageNet
rm -f $tinyImageNet


# Preprocessing the dataset
echo "==========================================="
echo "=           Dataset Preprocessing         ="
echo "==========================================="
#python setup/data_preprocess.py
