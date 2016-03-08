require 'image'
require 'lfs'
require 'nn'
require 'optim'
require 'torch'
require 'cunn'
require 'cudnn'
require 'cutorch'


local dtype = 'torch.CudaTensor'

local Dataset = require('dataLoader')
local createModel = require('models')

print('Loading training dataset...')
train = Dataset:create(train_table_path)
print('Loading validating dataset...')
val = Dataset:create(val_table_path)


print('Loading the model...')
model, crit = createModel(4)()


local X_train = train.data
local y_train = train.label
local X_val = val.data
local y_val = val.label

weights , grad_weights = net:getParameters()

local batch_size=100
local alpha=0.001
local num_epoch=40




local function f(w)
	assert(w == weights)
        X_batch, y_batch = train:getBatch(batch_size, true, .4, true)
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
        if i%10 == 0 then
	  print(string.format("iteration:%d",i))
        end
	local state = {learningRate = alpha}
	optim.adam(f, weights, state)
	if ((i%500)==0) then
		local trainInd=torch.LongTensor(5000):random(100000)
                local X_batchT = X_train:index(1,trainInd)
                local y_batchT = y_train:index(1,trainInd)

                local train_acc=eval(X_batchT,y_batchT,batch_size)
		local l=f(weights)
		local acc =eval(X_val,y_val,batch_size)	
		--local train_acc =eval(X_train,y_train,batch_size)	
		--local train_acc = 0
    
		print(string.format("Iteration=%d Loss=%f  Validation Accuracy=%f, Training Accuracy=%f",i,l,acc, train_acc))
		if (acc > best_acc) then
                        torch.save('net_best.t7',net)
                        best_acc = acc
                end

		log= log .. string.format("Iteration=%d Loss=%f  Validation Accuracy=%f, Training Accuracy=%f\n",i,l,acc,train_acc)
	end
	if(i% (100000 / batch_size) ==0) then 
		print(string.format("epoch = %d",epoch))
		epoch = epoch + 1
		alpha = alpha * 0.99
	end

end
log_file = io.open("log.txt","w")
io.output(log_file)
io.write(log)
io.close(log_file)

