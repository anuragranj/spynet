-- Copyright 2016 Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.  
-- By using this software you agree to the terms of the license file 
-- in the root folder.
-- For commercial use, please contact ps-license@tue.mpg.de.

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

local up2 =  nn.Sequential():add(nn.Transpose({2,3},{3,4})):add(nn.ScaleBHWD(2)):add(nn.Transpose({3,4},{2,3})):cuda()
local up3 =  nn.Sequential():add(nn.Transpose({2,3},{3,4})):add(nn.ScaleBHWD(2)):add(nn.Transpose({3,4},{2,3})):cuda()
local up4 =  nn.Sequential():add(nn.Transpose({2,3},{3,4})):add(nn.ScaleBHWD(2)):add(nn.Transpose({3,4},{2,3})):cuda()
local up5 =  nn.Sequential():add(nn.Transpose({2,3},{3,4})):add(nn.ScaleBHWD(2)):add(nn.Transpose({3,4},{2,3})):cuda()

local warpmodel2 = createWarpModel():cuda()
local warpmodel3 = createWarpModel():cuda()
local warpmodel4 = createWarpModel():cuda()
local warpmodel5 = createWarpModel():cuda()

down2:evaluate()
down3:evaluate()
down4:evaluate()
down5:evaluate()

up2:evaluate()
up3:evaluate()
up4:evaluate()
up5:evaluate()

warpmodel2:evaluate()
warpmodel3:evaluate()
warpmodel4:evaluate()
warpmodel5:evaluate()

-------------------------------------------------
local  modelL0, modelL1, modelL2, modelL3, modelL4, modelL5
local modelL1path, modelL2path, modelL3path, modelL4path, modelL5path

modelL1path = paths.concat('models', 'modelL1_3.t7')
modelL2path = paths.concat('models', 'modelL2_3.t7')
modelL3path = paths.concat('models', 'modelL3_3.t7')
modelL4path = paths.concat('models', 'modelL4_3.t7')
modelL5path = paths.concat('models', 'modelL5_3.t7')

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

local function getTrainValidationSplits(path)
   local numSamples = sys.fexecute( "ls " .. opt.data .. "| wc -l")/3
   local ff = torch.DiskFile(path, 'r')
   local trainValidationSamples = torch.IntTensor(numSamples)
   ff:readInt(trainValidationSamples:storage())
   ff:close()

   local train_samples = trainValidationSamples:eq(1):nonzero()
   local validation_samples = trainValidationSamples:eq(2):nonzero()

   return train_samples, validation_samples
  -- body
end
M.getTrainValidationSplits = getTrainValidationSplits

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

  local _flowappend = torch.zeros(opt.batchSize, 2, h, w):cuda()
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

local function getRawData(id)
   local path1 = paths.concat(opt.data, (string.format("%05i", id) .."_img1.ppm"))
   local path2 = paths.concat(opt.data, (string.format("%05i", id) .."_img2.ppm"))
   
   local img1 = loadImage(path1)
   local img2 = loadImage(path2)
   
   local pathF = paths.concat(opt.data, (string.format("%05i", id) .."_flow.flo"))
   local flow = loadFlow(pathF)

   return img1, img2, flow
end
M.getRawData = getRawData

local testHook = function(id)
   local path1 = paths.concat(opt.data, (string.format("%05i", id) .."_img1.ppm"))
   local path2 = paths.concat(opt.data, (string.format("%05i", id) .."_img2.ppm"))
   
   local img1 = loadImage(path1)
   local img2 = loadImage(path2)
   local images = torch.cat(img1, img2, 1)
   
   local pathF = paths.concat(opt.data, (string.format("%05i", id) .."_flow.flo"))
   local flow = loadFlow(pathF)
   
   images = TF.ColorNormalize(meanstd)(images)
   return images, flow
end
M.testHook = testHook

return M
