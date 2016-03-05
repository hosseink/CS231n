require 'image'
require 'lfs'
require 'nn'
require 'optim'
require 'torch'
require 'cunn'
require 'cutorch'

dataset_dir = '../datasets/tiny-imagenet-200/'
train_input = dataset_dir .. 'train/'
val_input = dataset_dir .. 'val/'
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



-- Defining The Model
net = nn.Sequential()
net:add (nn.SpatialConvolution(3,32,3,3))
net:add (nn.ReLU())

net:add (nn.SpatialConvolution(32,32,3,3))
net:add (nn.ReLU())

net:add (nn.View(32*60*60))
net:add (nn.Linear(32*60*60,200))
net:add (nn.ReLU())

net:add (nn.Linear(200,200))
net:type(dtype)

weights , grad_weights = net:getParameters()

-- Loss Function
crit = nn.CrossEntropyCriterion()
crit:type(dtype)

print("Network is Loaded!")



local function f(w)
	assert(w == weights)
        local r = math.random(0,10)
	local X_batch = X_train[{{1,100},{},{},{}}]:clone()
	local y_batch = y_train[{{1,100}}]:clone()
	local x = X_batch:type(dtype)
	local y = y_batch:type(dtype)
	
	local scores = net:forward(x)
	local loss = crit:forward(scores,y)

	grad_weights:zero()
	local dscores = crit:backward(scores,y)
	local dx = net:backward(x,dscores)

	return loss, grad_weights
end


print("Foward-Backward Function Loaded!")

for i=1,20 do
	local state = {learningRate = 1e-3}
	optim.adam(f, weights, state)
	local l=f(weights)
	print(l)
end


