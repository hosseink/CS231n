require 'image'
require 'lfs'
require 'nn'
require 'optim'
require 'torch'
require 'cunn'
require 'cudnn'
require 'cutorch'


dataset_dir = '../datasets/tiny-imagenet-200/'
train_input= dataset_dir .. 'train/'
val_input=dataset_dir .. 'val/'
local dtype = 'torch.CudaTensor'

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


local X_train = torch.Tensor(Nt,3,64,64)
local y_train = torch.Tensor(Nt)
local X_val = torch.Tensor(10000,3,64,64)
local y_val = torch.Tensor(10000)




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


print("Validation Set Loaded!")

cnt = 0
for folder in lfs.dir(train_input) do
        if string.sub(folder,1,1)=="n" then
                for file in lfs.dir(train_input .. folder .. "/images" ) do
                        if string.sub(file,1,1)=="n" then
				cnt = cnt+1
				y_train[cnt] = cDict[folder]
				ax=image.load(train_input .. folder .. "/images/" .. file)
				if ax:size(1)==3 then
                                	X_train[cnt] = ax 
                        	end
				if ax:size(1)==1 then
                                        X_train[{cnt,1}] = ax
					X_train[{cnt,2}] = ax
					X_train[{cnt,3}] = ax
                                end		
			end
                end
        end
end

print("Test Set Loaded!")

mean = {}
for i=1,3 do 
	mean[i] = X_train[{{},{i},{},{}}]:mean()
	X_train[{{},{i},{},{}}]:add(-mean[i])
	X_val[{{},{i},{},{}}]:add(-mean[i])
	print(mean[i])
end



local Convolution = nn.SpatialConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local iChannels
local Width

 local function shortcut(nInputPlane, nOutputPlane, stride)
      if nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Concat(2)
               :add(nn.Identity())
               :add(nn.MulConstant(0)))
      else
         return nn.Identity()
      end
   end



local function basicblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n
      Width = Width / stride

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1))
      s:add(SBatchNorm(n))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
end


local function layer(block, features, count, stride)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, i == 1 and stride or 1))
      end
      return s
end


net = nn.Sequential()
iChannels = 32
Width = 64

net:add(Convolution(3,32,5,5,1,1,2,2))
net:add(SBatchNorm(32))
net:add(ReLU(true))
net:add(layer(basicblock, 32, 2, 1))
net:add(layer(basicblock, 64, 2, 2))
net:add(layer(basicblock, 128, 2, 2))
net:add(layer(basicblock, 256, 2, 2))
--net:add(Max(2,2,2,2,0,0))
--net:add(layer(basicblock, 128, 1, 2))
net:add(Avg(2, 2, 1, 1))
net:add(nn.View(7*7*256))
net:add(nn.Linear(7*7*256, 200))

-- Defining The Model
--net = nn.Sequential()
--net:add (nn.SpatialConvolution(3,32,3,3,1,1,1,1))
--net:add (nn.SpatialBatchNormalization(32*64*64))
--net:add (nn.ReLU())
--net:add (nn.Dropout(0.5))

--net:add (nn.SpatialConvolution(32,32,3,3,1,1,1,1))
--net:add (nn.SpatialBatchNormalization(32*64*64))
--net:add (nn.ReLU())
--net:add (nn.Dropout(0.5))

--net:add (nn.View(32*64*64))
--net:add (nn.Linear(32*64*64,200))
--net:add (nn.BatchNormalization(400))
--net:add (nn.ReLU())

--net:add (nn.Linear(400,200))
net:type(dtype)

weights , grad_weights = net:getParameters()

-- Loss Function
crit = nn.CrossEntropyCriterion()
crit:type(dtype)

print("Network is Loaded!")

local batch_size=100
local alpha=0.001
local num_epoch=20




local function f(w)
	assert(w == weights)
	local batchInd=torch.LongTensor(batch_size):random(100000) 
	local X_batch = X_train:index(1,batchInd):clone()
	local y_batch = y_train:index(1,batchInd):clone()
	local x = X_batch:type(dtype)
	local y = y_batch:type(dtype)
	
	local scores = net:forward(x)
	local loss = crit:forward(scores,y)

	grad_weights:zero()
	local dscores = crit:backward(scores,y)
	local dx = net:backward(x,dscores)

	return loss, grad_weights
end


local function eval(x,y,batchSize)
	local NVal = x:size(1)
	local NBatch = NVal / batchSize
	local y_pred = torch.Tensor(NVal)
	local correct = 0
	for i=1,NBatch do 
		--print(string.format("val iter:%d",i))
		local s = (i-1)*batchSize +1
		local e = i*batchSize  
		local xg= x[{{s,e},{},{},{}}]:clone():type(dtype)
		local yg = y[{{s,e}}]:clone():type(dtype)
		--local xg = X_batch:type(dtype)
		--local yg = y_batch:type(dtype)
		net:evaluate()
		local scores = net:forward(xg)
		net:training()
		maxs, indices = torch.max(scores, 2)
		correct = correct + (torch.eq(yg,indices):sum())
	end
	return (100 * correct / NVal)	
end
	



print("Foward-Backward Function Loaded!")


local num_iter = num_epoch * 100000 / batch_size
local epoch=1
local best_acc=0
local log=string.format("alpha=%f batch size=%d num epoch=%d\n",alpha,batch_size,num_epoch)

for i=1,num_iter do
	--print(string.format("iteration:%d",i))
	local state = {learningRate = alpha}
	optim.adam(f, weights, state)
	if ((i%200)==0) then
		--local valInd=torch.LongTensor(1000):random(10000)
	        --local X_valbatch = X_val:index(1,valInd):clone()
        	--local y_valbatch = y_val:index(1,valInd):clone() 
		local l=f(weights)
		local acc =eval(X_val,y_val,batch_size)	
		print(string.format("Iteration=%d Loss=%f  Validation Accuracy=%f",i,l,acc))
		if (acc > best_acc) then
                        torch.save('net_best.t7',net)
                        best_acc = acc
                end

		log= log .. string.format("Iteration=%d Loss=%f  Validation Accuracy=%f\n",i,l,acc)
	end
	if(i% (100000 / batch_size) ==0) then 
		print(string.format("epoch = %d",epoch))
		epoch = epoch + 1
		alpha = alpha * 0.97
	end

end
log_file = io.open("log.txt","w")
io.output(log_file)
io.write(log)
io.close(log_file)

