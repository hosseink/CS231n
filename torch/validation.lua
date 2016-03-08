require 'image'
require 'lfs'
require 'nn'
require 'optim'
require 'torch'
require 'cunn'
require 'cudnn'
require 'cutorch'



train_input='/home/ubuntu/CS231n/datasets/tiny-imagenet-200/train/'
val_input='/home/ubuntu/CS231n/datasets/tiny-imagenet-200/val/'
local dtype = 'torch.CudaTensor'

local Dataset = require('dataLoader')

print('Loading validating dataset...')
val = Dataset:create(val_table_path)

local X_val = torch.Tensor(10000,3,64,64)
local y_val = torch.Tensor(10000)

local Nt = 0
local Nc = 0
cDict={}
for folder in lfs.dir(train_input) do
        if string.sub(folder,1,1)=="n" then
                Nc = Nc+1
                cDict[folder] = Nc
                for file in lfs.dir(train_input .. folder .. "/images" ) do
                        if string.sub(file,1,1)=="n" then
                                Nt = Nt + 1
                        end
                end
        end
end



local cnt=0
for line in io.lines(val_input .. "val_annotations.txt") do
      fName=string.format("val_%d.JPEG",cnt)
      cnt = cnt + 1
      y_val[cnt] = cDict[string.sub(line,string.find(line,"n%d%d%d%d%d%d%d%d"))]
      ax=image.load(val_input .. "/images/" .. fName)
      if ax:size(1)==3 then
              X_val[cnt] = ax
      end
      if ax:size(1)==1 then
              X_val[{cnt,1}] = ax
              X_val[{cnt,2}] = ax
              X_val[{cnt,3}] = ax
      end

end

mean={}
mean[1]=0.48023694334534
mean[2]=0.44806703677395
mean[3]=0.3975036419419


for i=1,3 do
	X_val[{{},{i},{},{}}]:add(-mean[i])
end

X_val = val.data
y_val = val.label
print("Validation Set Loaded!")

local net= torch.load('/home/ubuntu/CS231n/torch/net_best.t7')
net:type(dtype)

local function eval(x,y,batchSize)
        local NVal = x:size(1)
        local NBatch = NVal / batchSize
        local y_pred = torch.Tensor(NVal)
        local correct = 0
        for i=1,NBatch do
                print(math.floor(100 * i /NBatch) .. '%')
                local s = (i-1)*batchSize +1
                local e = i*batchSize
                local X_batch = x[{{s,e},{},{},{}}]:clone()
                local y_batch = y[{{s,e}}]:clone()
                
	        local yg = y_batch:type(dtype)
	        net:evaluate()
		local scores = torch.Tensor(batchSize, 200):zero():type(dtype)
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
		correct = correct + (torch.eq(yg,indF):sum())
        end
        return (100 * correct / NVal)
end


print(string.format("Validation Accuracy: %f",eval(X_val,y_val,100)))






