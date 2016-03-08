require 'image'
require 'lfs'
require 'nn'
require 'optim'
require 'torch'
require 'cunn'
require 'cudnn'
require 'cutorch'


--[[
currdir = lfs.currentdir()
idx = string.find(currdir, '/torch')
parentdir = string.sub(currdir, 1, idx)
parentdir = currdir:match("@?(.*/)")

print(parentdir)

local str = debug.getinfo(1, "S").source:sub(2)
print(str)
str = debug.getinfo(1).source:match("@?(.*/)")
print(str)
print(str:match("(.*/)"))
dataset_dir = parentdir .. 'datasets/tables/'
--]]

local dtype = 'torch.CudaTensor'
channel_mean = {0.48023694334534, 0.44806703677395, 0.3975036419419}

local function BuildArray(table)
  local arr = {}
  for v in table do
    arr[#arr + 1] = v
  end
  return arr
end

dataset_dir = '/home/ubuntu/CS231n/datasets/tables/'
train_table_path = dataset_dir .. 'train_table.txt'
val_table_path = dataset_dir .. 'val_table.txt'
test_table_path = dataset_dir .. 'test_table.txt'
words_table_path = dataset_dir .. 'words.txt'


num_of_classes = 200
num_of_training = 100000
num_of_val = 10000
num_of_test = 10000
width, height = 64, 64
crop_ratio = .7

local function ReadTable(table_path)
  local table = {}
  size = 1
  for line in io.lines(table_path) do
    local tokens = BuildArray(string.gmatch(line, "[^%s]+")) 
    table[#table + 1] = {tokens[1], tokens[2]}
    size = size + 1
  end
  return size-1, table
end

local function LoadData(table_path, size)
  local data = torch.Tensor(size, 3, 64, 64)
  local label = torch.Tensor(size)
  i = 1
  for line in io.lines(table_path) do
    if(100 * i % size) == 0 then
      print(i*100 / size .. '%')
    end
    local tokens = BuildArray(string.gmatch(line, "[^%s]+")) 
    label[i] = tokens[2]
    img = image.load(tokens[1])
    if img:size(1) == 3 then
      data[i] = img
    else
      data[{i, 1}] = img 
      data[{i, 2}] = img
      data[{i, 3}] = img
    end 
    i = i + 1
  end
  data[{{}, {1}, {}, {}}]:add(- channel_mean[1])
  data[{{}, {2}, {}, {}}]:add(- channel_mean[2])
  data[{{}, {3}, {}, {}}]:add(- channel_mean[3])
  return data, label
end


Dataset = {}                                  
function Dataset:create(path)                
  dataset = {}               
  size = ReadTable(path) 
  data, label = LoadData(path, size)
  dataset.size = size
  dataset.data = data
  dataset.label = label
  self.__index = self 
  return setmetatable(dataset, self)
end

crop_width = math.floor(width * crop_ratio)
crop_height = math.floor(height * crop_ratio)
local crop_indices = {[1] = {0,0,crop_width, crop_height},
                      [2] = {width - crop_width, height - crop_height, width, height},
		      [3] = {0, height - crop_height, crop_width, height},
                      [4] = {width - crop_width, 0 , width, crop_height}}

function Dataset:getBatch(batch_size, withAugmentation, p)
  local batchInd=torch.LongTensor(batch_size):random(self.size) 
  local X = self.data:index(1,batchInd):clone()
  local y = self.label:index(1,batchInd):clone()
 
  if withAugmentation then
    for i = 1, batch_size do
      local x = X[i]:clone()
      if math.random() < .5 then
        x = image.hflip(x)
      end
      local q = math.random()
      if q < p then
        crop = crop_indices[math.floor(4 * q / p) + 1]
        x = image.crop(x, crop[1], crop[2], crop[3], crop[4])  
        x = image.scale(x, width) 
      end
      X[i] = x
    end 
  end
  return X, y
end

function getDict()
  cDict = {}
  for line in io.lines(words_table_path) do
    local tokens = BuildArray(string.gmatch(line, "[^%s]+")) 
    cDict[tokens[1]] = tokens[2]
  end
  return cDict
end

cDict = getDict()

function Dataset:getAugmentation(X, aug_type)
  batch_size = X:size(1)
  OUT = torch.Tensor(X:size(1), X:size(2), X:size(3), X:size(4))
  for i = 1, batch_size do
    local x = X[i]:clone()
    if aug_type < 5 then
        x = image.hflip(x)
    end   
    if aug_type%5 ~= 0 then
      crop = crop_indices[aug_type%5]
      x = image.crop(x, crop[1], crop[2], crop[3], crop[4])  
      x = image.scale(x, width) 
    end
    OUT[i] = x
  end
  return OUT
end

return Dataset
