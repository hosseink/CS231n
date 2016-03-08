local nn = require 'nn'
require 'cunn'
local dtype = 'torch.CudaTensor'



local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

-------------------------------------------------------------------
local function shortcut(nInputPlane, nOutputPlane, stride, useConv)
  if useConv then
    -- 1x1 convolution
    return nn.Sequential()
      :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
      :add(SBatchNorm(nOutputPlane))
  elseif nInputPlane ~= nOutputPlane then
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

------------------------------------------------------------------
local iChannel

local function basicblock(n, stride)
  local nInputPlane = iChannels
  iChannels = n

  local s = nn.Sequential()
  s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
  s:add(SBatchNorm(n))
  s:add(ReLU(true))
  s:add(Convolution(n,n,3,3,1,1,1,1))
  s:add(SBatchNorm(n))

  return nn.Sequential()
       :add(nn.ConcatTable()
       :add(s)
       :add(shortcut(nInputPlane, n, stride, false)))
       :add(nn.CAddTable(true))
       :add(ReLU(true))
end
-------------------------------------------------------------------
local function layer(block, features, count, stride)
  local s = nn.Sequential()
  for i=1,count do
    s:add(block(features, i == 1 and stride or 1))
  end
  return s
end
-------------------------------------------------------------------
------------------------------MODEL 1------------------------------
-------------------------------------------------------------------
local function createModel1()

  net = nn.Sequential()
  iChannels = 64

  net:add(Convolution(3,64,3,3,1,1,1,1))
  net:add(SBatchNorm(64*64*64))
  net:add(ReLU(true))
  net:add(Max(2,2,2,2,0,0))
  net:add(layer(basicblock, 64, 2, 1))
  net:add(layer(basicblock, 128, 2, 2))
  net:add(layer(basicblock, 256, 2, 2))
  net:add(layer(basicblock, 512, 2, 2))
  net:add(Avg(2, 2, 1, 1))
  net:add(nn.View(3*3*512))
  net:add(nn.Linear(3*3*512, 200))

  net:type(dtype)

  -- Loss Function
  crit = nn.CrossEntropyCriterion()
  crit:type(dtype)

  return net, crit
end

-------------------------------------------------------------------
------------------------------MODEL 2------------------------------
-------------------------------------------------------------------
local function createModel2()

  net = nn.Sequential()
  iChannels = 64

  net:add(Convolution(3,64,3,3,1,1,1,1))
  net:add(Convolution(64,64,3,3,1,1,1,1))
  net:add(SBatchNorm(64))
  net:add(ReLU(true))
  net:add(Max(2,2,2,2,0,0))
  net:add(layer(basicblock, 64, 2, 1))
  net:add(layer(basicblock, 128, 3, 2))
  net:add(layer(basicblock, 256, 4, 2))
  net:add(layer(basicblock, 512, 2, 2))
  net:add(Avg(2, 2, 1, 1))
  net:add(nn.View(3*3*512))
  net:add(nn.Linear(3*3*512, 200))

  net:type(dtype)

  -- Loss Function
  crit = nn.CrossEntropyCriterion()
  crit:type(dtype)

  return net, crit
end
-------------------------------------------------------------------


-------------------------------------------------------------------
local function createModel(modelNumber)
  if modelNumber == 1 then
    return createModel1
  elseif modelNumber == 2 then
    return createModel2
  elseif modelNumber == 3 then
    return createModel3
  elseif modelNumber == 4 then
    return createModel4
  elseif modelNumber == 5 then
    return createModel5
  elseif modelNumber == 6 then
    return createModel6
  elseif modelNumber == 7 then
    return createModel7
  elseif modelNumber == 8 then
    return createModel8
  end
end

return createModel
