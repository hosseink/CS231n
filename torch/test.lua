require 'image'
require 'lfs'
require 'nn'
require 'optim'
require 'torch'
require 'cunn'
require 'cudnn'
require 'cutorch'


train_input='/home/ubuntu/CS231n/datasets/tiny-imagenet-200/train/'
test_input='/home/ubuntu/CS231n/datasets/tiny-imagenet-200/test/'
local dtype = 'torch.CudaTensor'

local Dataset = require('dataLoader')

print('Loading validating dataset...')
val = Dataset:create(val_table_path)


local X_val = torch.Tensor(10000,3,64,64)

local Nt = 0
local Nc = 0
--[[
cDict={}
for folder in lfs.dir(train_input) do
        if string.sub(folder,1,1)=="n" then
                Nc = Nc+1
                cDict[Nc] = folder
        end
end
--]]
--print(cDict[1 .. ''])
testName={}
local cnt=0
for fName in lfs.dir(test_input .. "/images/") do
    if string.sub(fName,1,1)=="t" then
      cnt = cnt +1
      testName[cnt]=fName
      ax=image.load(test_input .. "/images/" .. fName)
      if ax:size(1)==3 then
              X_val[cnt] = ax
      end
      if ax:size(1)==1 then
              X_val[{cnt,1}] = ax
              X_val[{cnt,2}] = ax
              X_val[{cnt,3}] = ax
      end
    end
end

mean={}
mean[1]=0.48023694334534
mean[2]=0.44806703677395
mean[3]=0.3975036419419


for i=1,3 do
	X_val[{{},{i},{},{}}]:add(-mean[i])
end


print("Test Set Loaded!")

local net= torch.load('/home/ubuntu/CS231n/torch/net_best.t7')
net:type(dtype)

x=X_val
batchSize=200



local NVal = x:size(1)
local NBatch = NVal / batchSize
local y_pred = torch.Tensor(NVal):type(dtype)
for i=1,NBatch do
        print(math.floor(100 * i /NBatch) .. '%')
	local s = (i-1)*batchSize +1
	local e = i*batchSize
	local X_batch = x[{{s,e},{},{},{}}]:clone()
	net:evaluate()
	scores = torch.Tensor(batchSize, 200):zero():type(dtype)
	local majScores = torch.Tensor(batchSize,200):zero():type(dtype)
	for j = 0, 9 do
		 local xgi = val:getAugmentation(X_batch, j):type(dtype)   
		 scores = net:forward(xgi:type(dtype))
		 maxs, indices = torch.max(scores, 2)
		 for k=1,batchSize do
			majScores[{k,indices[k][1]}] = majScores[{k,indices[k][1]}] + 1		
		 end
	end
	mm,indF = torch.max(majScores,2)
	net:training()
	y_pred[{{s,e}}]= indF:clone()
end

print("Prediction Completed!")
 
result = io.open("sadegh.txt","w")
io.output(result)
for i=1,NVal do
	io.write(testName[i] .. " " .. cDict[y_pred[i] .. ''] .. "\n")
end
io.close(result)







