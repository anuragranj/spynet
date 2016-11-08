require 'image'
local TF = require 'transforms'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'stn'
require 'spy'
local flowX = require 'flowExtensions'

local M = {}

local eps = 1e-6
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

local mean = meanstd.mean
local std = meanstd.std
------------------------------------------
local function createWarpModel()
  local imgData = nn.Identity()()
  local floData = nn.Identity()()

  local imgOut = nn.Transpose({2,3},{3,4})(imgData)
  local floOut = nn.Transpose({2,3},{3,4})(floData)

  local warpImOut = nn.Transpose({3,4},{2,3})(nn.BilinearSamplerBHWD()({imgOut, floOut}))
  local model = nn.gModule({imgData, floData}, {warpImOut})

  return model
end

local down2 = nn.SpatialAveragePooling(2,2,2,2):cuda()
local down3 = nn.SpatialAveragePooling(2,2,2,2):cuda()
local down4 = nn.SpatialAveragePooling(2,2,2,2):cuda()
local down5 = nn.SpatialAveragePooling(2,2,2,2):cuda()
local down6 = nn.SpatialAveragePooling(2,2,2,2):cuda()

local up2 =  nn.Sequential():add(nn.Transpose({2,3},{3,4})):add(nn.ScaleBHWD(2)):add(nn.Transpose({3,4},{2,3})):cuda()
local up3 =  nn.Sequential():add(nn.Transpose({2,3},{3,4})):add(nn.ScaleBHWD(2)):add(nn.Transpose({3,4},{2,3})):cuda()
local up4 =  nn.Sequential():add(nn.Transpose({2,3},{3,4})):add(nn.ScaleBHWD(2)):add(nn.Transpose({3,4},{2,3})):cuda()
local up5 =  nn.Sequential():add(nn.Transpose({2,3},{3,4})):add(nn.ScaleBHWD(2)):add(nn.Transpose({3,4},{2,3})):cuda()
local up6 =  nn.Sequential():add(nn.Transpose({2,3},{3,4})):add(nn.ScaleBHWD(2)):add(nn.Transpose({3,4},{2,3})):cuda()

local warpmodel2 = createWarpModel():cuda()
local warpmodel3 = createWarpModel():cuda()
local warpmodel4 = createWarpModel():cuda()
local warpmodel5 = createWarpModel():cuda()
local warpmodel6 = createWarpModel():cuda()

down2:evaluate()
down3:evaluate()
down4:evaluate()
down5:evaluate()
down6:evaluate()

up2:evaluate()
up3:evaluate()
up4:evaluate()
up5:evaluate()
up6:evaluate()

warpmodel2:evaluate()
warpmodel3:evaluate()
warpmodel4:evaluate()
warpmodel5:evaluate()
warpmodel6:evaluate()

-------------------------------------------------
local modelL1, modelL2, modelL3, modelL4, modelL5, modelL6
local modelL1path, modelL2path, modelL3path, modelL4path, modelL5path, modelL6path

local function loadImage(path)
   local input = image.load(path, 3, 'float')
   return input
end
M.loadImage = loadImage

local function loadFlow(filename)
  TAG_FLOAT = 202021.25 
  local ff = torch.DiskFile(filename):binary()
  local tag = ff:readFloat()
  if tag ~= TAG_FLOAT then
    xerror('unable to read '..filename..
     ' perhaps bigendian error','readflo()')
  end
   
  local w = ff:readInt()
  local h = ff:readInt()
  local nbands = 2
  local tf = torch.FloatTensor(h, w, nbands)
  ff:readFloat(tf:storage())
  ff:close()

  local flow = tf:permute(3,1,2)  
  return flow
end
M.loadFlow = loadFlow


local function computeInitFlowL1(imagesL1)
  local h = imagesL1:size(3)
  local w = imagesL1:size(4)
  local batchSize = imagesL1:size(1)

  local _flowappend = torch.zeros(batchSize, 2, h, w):cuda()
  local images_in = torch.cat(imagesL1, _flowappend, 2)

  local flow_est = modelL1:forward(images_in)
  return flow_est
end
M.computeInitFlowL1 = computeInitFlowL1

local function computeInitFlowL2(imagesL2)
  local imagesL1 = down2:forward(imagesL2:clone())  
  local _flowappend = up2:forward(computeInitFlowL1(imagesL1))*2    
  local _img2 = imagesL2[{{},{4,6},{},{}}]
  imagesL2[{{},{4,6},{},{}}]:copy(warpmodel2:forward({_img2, _flowappend}))

  local images_in = torch.cat(imagesL2, _flowappend, 2)
  
  local  flow_est = modelL2:forward(images_in)
  return flow_est:add(_flowappend)
end
M.computeInitFlowL2 = computeInitFlowL2

local function computeInitFlowL3(imagesL3)
  local imagesL2 = down3:forward(imagesL3:clone())
  local _flowappend = up3:forward(computeInitFlowL2(imagesL2))*2  
  local _img2 = imagesL3[{{},{4,6},{},{}}]
  imagesL3[{{},{4,6},{},{}}]:copy(warpmodel3:forward({_img2, _flowappend}))

  local images_in = torch.cat(imagesL3, _flowappend, 2)
  
  local  flow_est = modelL3:forward(images_in)
  return flow_est:add(_flowappend)
end
M.computeInitFlowL3 = computeInitFlowL3

local  function computeInitFlowL4(imagesL4)
  local imagesL3 = down4:forward(imagesL4)
  local _flowappend = up4:forward(computeInitFlowL3(imagesL3))*2  
  local _img2 = imagesL4[{{},{4,6},{},{}}]
  imagesL4[{{},{4,6},{},{}}]:copy(warpmodel4:forward({_img2, _flowappend}))

  local images_in = torch.cat(imagesL4, _flowappend, 2)
  
  local  flow_est = modelL4:forward(images_in)
  return flow_est:add(_flowappend)
end
M.computeInitFlowL4 = computeInitFlowL4

local function computeInitFlowL5(imagesL5)  
  local imagesL4 = down5:forward(imagesL5)
  local _flowappend = up5:forward(computeInitFlowL4(imagesL4))*2  
  
  local _img2 = imagesL5[{{},{4,6},{},{}}]
  imagesL5[{{},{4,6},{},{}}]:copy(warpmodel5:forward({_img2, _flowappend}))

  local images_in = torch.cat(imagesL5, _flowappend, 2)
  
  local  flow_est = modelL5:forward(images_in)
  return flow_est:add(_flowappend)
end
M.computeInitFlowL5 = computeInitFlowL5

local function computeInitFlowL6(imagesL6)  
  local imagesL5 = down6:forward(imagesL6)
  local _flowappend = up6:forward(computeInitFlowL5(imagesL5))*2  
  
  local _img2 = imagesL6[{{},{4,6},{},{}}]
  imagesL6[{{},{4,6},{},{}}]:copy(warpmodel6:forward({_img2, _flowappend}))

  local images_in = torch.cat(imagesL6, _flowappend, 2)
  
  local  flow_est = modelL6:forward(images_in)
  return flow_est:add(_flowappend)
end
M.computeInitFlowL6 = computeInitFlowL6


local function setup(width, height, opt)
  opt = opt or "sintelFinal"
  local len = math.max(width, height)
  local computeFlow
  local level

  if len <= 32 then
    computeFlow = computeInitFlowL1
    level = 1
  elseif len <= 64 then
    computeFlow = computeInitFlowL2
    level = 2
  elseif len <= 128 then
    computeFlow = computeInitFlowL3
    level = 3
  elseif len <= 256 then
    computeFlow = computeInitFlowL4
    level = 4
  elseif len <= 512 then
    computeFlow = computeInitFlowL5
    level = 5
  elseif len <= 1024 then
    computeFlow = computeInitFlowL6
    level = 6
  else
    error("Only image size <= 1024 supported. Next release will have full support.")
  end

  if opt=="sintelFinal" then
    modelL1path = paths.concat('models', 'modelL1_F.t7')
    modelL2path = paths.concat('models', 'modelL2_F.t7')
    modelL3path = paths.concat('models', 'modelL3_F.t7')
    modelL4path = paths.concat('models', 'modelL4_F.t7')
    modelL5path = paths.concat('models', 'modelL5_F.t7')
    modelL6path = paths.concat('models', 'modelL6_F.t7')
  end

  if opt=="sintelClean" then
    modelL1path = paths.concat('models', 'modelL1_C.t7')
    modelL2path = paths.concat('models', 'modelL2_C.t7')
    modelL3path = paths.concat('models', 'modelL3_C.t7')
    modelL4path = paths.concat('models', 'modelL4_C.t7')
    modelL5path = paths.concat('models', 'modelL5_C.t7')
    modelL6path = paths.concat('models', 'modelL6_C.t7')
  end

  if opt=="chairsClean" then
    modelL1path = paths.concat('models', 'modelL1_4.t7')
    modelL2path = paths.concat('models', 'modelL2_4.t7')
    modelL3path = paths.concat('models', 'modelL3_4.t7')
    modelL4path = paths.concat('models', 'modelL4_4.t7')
    modelL5path = paths.concat('models', 'modelL5_4.t7')
    modelL6path = paths.concat('models', 'modelL5_4.t7')
  end

  if opt=="chairsFinal" then
    modelL1path = paths.concat('models', 'modelL1_3.t7')
    modelL2path = paths.concat('models', 'modelL2_3.t7')
    modelL3path = paths.concat('models', 'modelL3_3.t7')
    modelL4path = paths.concat('models', 'modelL4_3.t7')
    modelL5path = paths.concat('models', 'modelL5_3.t7')
    modelL6path = paths.concat('models', 'modelL5_3.t7')
  end

  if level>0 then
    modelL1 = torch.load(modelL1path)
    if torch.type(modelL1) == 'nn.DataParallelTable' then
       modelL1 = modelL1:get(1)
    end
    modelL1:evaluate()
  end

  if level>1 then
    modelL2 = torch.load(modelL2path)
    if torch.type(modelL2) == 'nn.DataParallelTable' then
       modelL2 = modelL2:get(1)
    end
    modelL2:evaluate()
  end

  if level>2 then
    modelL3 = torch.load(modelL3path)
    if torch.type(modelL3) == 'nn.DataParallelTable' then
       modelL3 = modelL3:get(1)
    end
    modelL3:evaluate()
  end

  if level>3 then
    modelL4 = torch.load(modelL4path)
    if torch.type(modelL4) == 'nn.DataParallelTable' then
      modelL4 = modelL4:get(1)
    end
    modelL4:evaluate()
  end

  if level>4 then
    modelL5 = torch.load(modelL5path)
    if torch.type(modelL5) == 'nn.DataParallelTable' then
      modelL5 = modelL5:get(1)
    end
    modelL5:evaluate()
  end

  if level>5 then
    modelL6 = torch.load(modelL6path)
    if torch.type(modelL6) == 'nn.DataParallelTable' then
      modelL6 = modelL6:get(1)
    end
    modelL6:evaluate()
  end
  
  return computeFlow
end
M.setup = setup

local function DeAdjustFlow(flow, h, w)
  local sc_h = h/flow:size(2)
  local sc_w = w/flow:size(3)
  flow = image.scale(flow, w, h, 'simple')
  flow[2] = flow[2]*sc_h
  flow[1] = flow[1]*sc_w

  return flow
end
M.DeAdjustFlow = DeAdjustFlow

local easyComputeFlow = function(im1, im2)
  local imgs = torch.cat(im1, im2, 1)

  local width = imgs:size(3)
  local height = imgs:size(2)
  
  local fineWidth, fineHeight
  
  if width%32 == 0 then
    fineWidth = width
  else
    fineWidth = width + 32 - math.fmod(width, 32)
  end

  if height%32 == 0 then
    fineHeight = height
  else
    fineHeight = height + 32 - math.fmod(height, 32)
  end  
       
  imgs = image.scale(imgs, fineWidth, fineHeight)

  local len = math.max(fineWidth, fineHeight)
  local computeFlow
  
  if len <= 32 then
    computeFlow = computeInitFlowL1
  elseif len <= 64 then
    computeFlow = computeInitFlowL2
  elseif len <= 128 then
    computeFlow = computeInitFlowL3
  elseif len <= 256 then
    computeFlow = computeInitFlowL4
  elseif len <= 512 then
    computeFlow = computeInitFlowL5
  else
    computeFlow = computeInitFlowL6
  end

  imgs = imgs:resize(1,6,fineHeight,fineWidth):cuda()
  local flow_est = computeFlow(imgs)

  flow_est = flow_est:squeeze():float()
  flow_est = DeAdjustFlow(flow_est, height, width)

  return flow_est

end

local function easy_setup()
  modelL1path = paths.concat('models', 'modelL1_F.t7')
  modelL2path = paths.concat('models', 'modelL2_F.t7')
  modelL3path = paths.concat('models', 'modelL3_F.t7')
  modelL4path = paths.concat('models', 'modelL4_F.t7')
  modelL5path = paths.concat('models', 'modelL5_F.t7')
  modelL6path = paths.concat('models', 'modelL6_F.t7')

  modelL1 = torch.load(modelL1path)
  if torch.type(modelL1) == 'nn.DataParallelTable' then
     modelL1 = modelL1:get(1)
  end
  modelL1:evaluate()

  modelL2 = torch.load(modelL2path)
  if torch.type(modelL2) == 'nn.DataParallelTable' then
     modelL2 = modelL2:get(1)
  end
  modelL2:evaluate()

  modelL3 = torch.load(modelL3path)
  if torch.type(modelL3) == 'nn.DataParallelTable' then
     modelL3 = modelL3:get(1)
  end
  modelL3:evaluate()

  modelL4 = torch.load(modelL4path)
  if torch.type(modelL4) == 'nn.DataParallelTable' then
    modelL4 = modelL4:get(1)
  end
  modelL4:evaluate()

  modelL5 = torch.load(modelL5path)
  if torch.type(modelL5) == 'nn.DataParallelTable' then
    modelL5 = modelL5:get(1)
  end
  modelL5:evaluate()

  modelL6 = torch.load(modelL6path)
  if torch.type(modelL6) == 'nn.DataParallelTable' then
    modelL6 = modelL6:get(1)
  end
  modelL6:evaluate()
  return easyComputeFlow
end
M.easy_setup = easy_setup



return M