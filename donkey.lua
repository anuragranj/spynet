-- Copyright 2016 Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.  
-- By using this software you agree to the terms of the license file 
-- in the root folder.
-- For commercial use, please contact ps-license@tue.mpg.de.

require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'stn'
require 'spy'

local flowX = require 'flowExtensions'
local TF = require 'transforms'

paths.dofile('dataset.lua')
paths.dofile('util.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
local eps = 1e-6
-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = paths.concat(opt.cache, 'trainCache.t7')
local testCache = paths.concat(opt.cache, 'testCache.t7')

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
-- Warping Function:
local function createWarpModel()
  local imgData = nn.Identity()()
  local floData = nn.Identity()()

  local imgOut = nn.Transpose({2,3},{3,4})(imgData)
  local floOut = nn.Transpose({2,3},{3,4})(floData)

  local warpImOut = nn.Transpose({3,4},{2,3})(nn.BilinearSamplerBHWD()({imgOut, floOut}))
  local model = nn.gModule({imgData, floData}, {warpImOut})

  return model
end

local modelL1, modelL2, modelL3, modelL4
local modelL1path, modelL2path, modelL3path, modelL4path
local down1, down2, down3, down4, up2, up3, up4
local warpmodel2, warpmodel3, warpmodel4

modelL1path = opt.L1
modelL2path = opt.L2
modelL3path = opt.L3
modelL4path = opt.L4

if opt.level > 1 then
   -- Load modelL1
   modelL1 = torch.load(modelL1path)
   if torch.type(modelL1) == 'nn.DataParallelTable' then
      modelL1 = modelL1:get(1)
   end
   modelL1:evaluate()
   down1 = nn.SpatialAveragePooling(2,2,2,2):cuda()
   down1:evaluate()
end

if opt.level > 2 then
-- Load modelL2
   modelL2 = torch.load(modelL2path)
   if torch.type(modelL2) == 'nn.DataParallelTable' then
      modelL2 = modelL2:get(1)
   end
   modelL2:evaluate()

   down2 = nn.SpatialAveragePooling(2,2,2,2):cuda()
   up2 =  nn.Sequential():add(nn.Transpose({2,3},{3,4})):add(nn.ScaleBHWD(2)):add(nn.Transpose({3,4},{2,3})):cuda()
   warpmodel2 = createWarpModel():cuda()

   down2:evaluate()
   up2:evaluate()
   warpmodel2:evaluate()
end

if opt.level > 3 then
   -- Load modelL3
   modelL3 = torch.load(modelL3path)
   if torch.type(modelL3) == 'nn.DataParallelTable' then
      modelL3 = modelL3:get(1)
   end
   modelL3:evaluate()

   down3 = nn.SpatialAveragePooling(2,2,2,2):cuda()
   up3 =  nn.Sequential():add(nn.Transpose({2,3},{3,4})):add(nn.ScaleBHWD(2)):add(nn.Transpose({3,4},{2,3})):cuda()
   warpmodel3 = createWarpModel():cuda()

   down3:evaluate()
   up3:evaluate()
   warpmodel3:evaluate()
end

if opt.level > 4 then
   -- Load modelL4
   modelL4 = torch.load(modelL4path)
   if torch.type(modelL4) == 'nn.DataParallelTable' then
      modelL4 = modelL4:get(1)
   end
   modelL4:evaluate()

   down4 = nn.SpatialAveragePooling(2,2,2,2):cuda()
   up4 =  nn.Sequential():add(nn.Transpose({2,3},{3,4})):add(nn.ScaleBHWD(2)):add(nn.Transpose({3,4},{2,3})):cuda()
   warpmodel4 = createWarpModel():cuda()

   down4:evaluate()
   up4:evaluate()
   warpmodel4:evaluate()
end

-- Check for existence of opt.data
if not os.execute('cd ' .. opt.data) then
   error(("could not chdir to '%s'"):format(opt.data))
end

local loadSize = opt.loadSize
local inputSize = {8, opt.fineHeight, opt.fineWidth}
local outputSize = {2, opt.fineHeight, opt.fineWidth}

local function getTrainValidationSplits(path)
   local numSamples = sys.fexecute( "ls " .. opt.data .. "| wc -l")/3
   local ff = torch.DiskFile(path, 'r')
   local trainValidationSamples = torch.IntTensor(numSamples)
   ff:readInt(trainValidationSamples:storage())
   ff:close()

   local train_samples = trainValidationSamples:eq(1):nonzero()
   local validation_samples = trainValidationSamples:eq(2):nonzero()

   return train_samples, validation_samples
end

local train_samples, validation_samples = getTrainValidationSplits(opt.trainValidationSplit)

local function loadImage(path)
   local input = image.load(path, 3, 'float')
   return input
end

local  function rotateFlow(flow, angle)
  local flow_rot = image.rotate(flow, angle)
  local fu = torch.mul(flow_rot[1], math.cos(-angle)) - torch.mul(flow_rot[2], math.sin(-angle)) 
  local fv = torch.mul(flow_rot[1], math.sin(-angle)) + torch.mul(flow_rot[2], math.cos(-angle))
  flow_rot[1]:copy(fu)
  flow_rot[2]:copy(fv)

  return flow_rot
end

local function scaleFlow(flow, height, width)
  -- scale the original flow to a flow of size height x width
  local sc = height/flow:size(2)
  assert(torch.abs(width/flow:size(3) - sc)<eps, 'Aspect ratio of output flow is not the same as input flow' )
  local flow_scaled = image.scale(flow, width, height)*sc
  return flow_scaled
end

local function computeInitFlowL1(imagesL1)
  local h = imagesL1:size(3)
  local w = imagesL1:size(4)
  local batchSize = imagesL1:size(1)

  local _flowappend = torch.zeros(batchSize, 2, h, w):cuda()
  local images_in = torch.cat(imagesL1, _flowappend, 2)

  local flow_est = modelL1:forward(images_in)
  return flow_est
end

local function computeInitFlowL2(imagesL2)
  local imagesL1 = down2:forward(imagesL2:clone())  
  local _flowappend = up2:forward(computeInitFlowL1(imagesL1))*2    
  local _img2 = imagesL2[{{},{4,6},{},{}}]
  imagesL2[{{},{4,6},{},{}}]:copy(warpmodel2:forward({_img2, _flowappend}))

  local images_in = torch.cat(imagesL2, _flowappend, 2)
  
  local  flow_est = modelL2:forward(images_in)
  return flow_est:add(_flowappend)
end

local function computeInitFlowL3(imagesL3)
  local imagesL2 = down3:forward(imagesL3:clone())
  local _flowappend = up3:forward(computeInitFlowL2(imagesL2))*2  
  local _img2 = imagesL3[{{},{4,6},{},{}}]
  imagesL3[{{},{4,6},{},{}}]:copy(warpmodel3:forward({_img2, _flowappend}))

  local images_in = torch.cat(imagesL3, _flowappend, 2)
  
  local  flow_est = modelL3:forward(images_in)
  return flow_est:add(_flowappend)
end

local  function computeInitFlowL4(imagesL4)
  local imagesL3 = down4:forward(imagesL4)
  local _flowappend = up4:forward(computeInitFlowL3(imagesL3))*2  
  local _img2 = imagesL4[{{},{4,6},{},{}}]
  imagesL4[{{},{4,6},{},{}}]:copy(warpmodel4:forward({_img2, _flowappend}))

  local images_in = torch.cat(imagesL4, _flowappend, 2)
  
  local  flow_est = modelL4:forward(images_in)
  return flow_est:add(_flowappend)
end

local function makeData(images, flows)
  local initFlow, flowDiffOutput
  local images_scaled = image.scale(images, opt.fineWidth, opt.fineHeight)
  
  if opt.level == 1 then
    initFlow = torch.zeros(2, opt.fineHeight, opt.fineWidth)
    flowDiffOutput = scaleFlow(flows, opt.fineHeight, opt.fineWidth) 

  elseif opt.level == 2 then
    local coarseImages = image.scale(images, opt.fineWidth/2, opt.fineHeight/2)
    initFlow = computeInitFlowL1(coarseImages:resize(1,coarseImages:size(1), 
              coarseImages:size(2), coarseImages:size(3)):cuda())
    initFlow = scaleFlow(initFlow:squeeze():float(), opt.fineHeight, opt.fineWidth)

    flowDiffOutput = scaleFlow(flows, opt.fineHeight, opt.fineWidth)
    flowDiffOutput = flowDiffOutput:add(flowDiffOutput, -1, initFlow)

  elseif opt.level == 3 then
    local coarseImages = image.scale(images, opt.fineWidth/2, opt.fineHeight/2)
    initFlow = computeInitFlowL1(coarseImages:resize(1,coarseImages:size(1), 
              coarseImages:size(2), coarseImages:size(3)):cuda())
    initFlow = scaleFlow(initFlow:squeeze():float(), opt.fineHeight, opt.fineWidth)

    flowDiffOutput = scaleFlow(flows, opt.fineHeight, opt.fineWidth)
    flowDiffOutput = flowDiffOutput:add(flowDiffOutput, -1, initFlow)

  elseif opt.level == 4 then
    local coarseImages = image.scale(images, opt.fineWidth/2, opt.fineHeight/2)
    initFlow = computeInitFlowL1(coarseImages:resize(1,coarseImages:size(1), 
              coarseImages:size(2), coarseImages:size(3)):cuda())
    initFlow = scaleFlow(initFlow:squeeze():float(), opt.fineHeight, opt.fineWidth)

    flowDiffOutput = scaleFlow(flows, opt.fineHeight, opt.fineWidth)
    flowDiffOutput = flowDiffOutput:add(flowDiffOutput, -1, initFlow)

  elseif opt.level == 5 then
    local coarseImages = image.scale(images, opt.fineWidth/2, opt.fineHeight/2)
    initFlow = computeInitFlowL1(coarseImages:resize(1,coarseImages:size(1), 
              coarseImages:size(2), coarseImages:size(3)):cuda())
    initFlow = scaleFlow(initFlow:squeeze():float(), opt.fineHeight, opt.fineWidth)

    flowDiffOutput = scaleFlow(flows, opt.fineHeight, opt.fineWidth)
    flowDiffOutput = flowDiffOutput:add(flowDiffOutput, -1, initFlow)

  end

  local _img2 = images_scaled[{{4,6},{},{}}]:clone()
  images_scaled[{{4,6},{},{}}]:copy(image.warp(_img2, initFlow:index(1, torch.LongTensor{2,1})))
  
  local imageFlowInputs = torch.cat(images_scaled, initFlow:float(), 1)
  return imageFlowInputs, flowDiffOutput
end


local function Preprocess()
  return TF.Compose{
    TF.ColorJitter({
      brightness = 0.4,
      contrast = 0.4,
      saturation = 0.4,
      }),
    TF.Lighting(0.1, pca.eigval, pca.eigvec),
    TF.ColorNormalize(meanstd),
    }
end


-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, id)
   collectgarbage()
   local path1 = paths.concat(opt.data, (string.format("%05i", id) .."_img1.ppm"))
   local path2 = paths.concat(opt.data, (string.format("%05i", id) .."_img2.ppm"))
   
   local img1 = loadImage(path1)
   local img2 = loadImage(path2)
   local images = torch.cat(img1, img2, 1)
   
   local pathF = paths.concat(opt.data, (string.format("%05i", id) .."_flow.flo"))
   local flow = flowX.loadFLO(pathF)

   local imagesOut, flowOut
  
   if opt.augment == 1 then
     -- Rotation [-0.3 0.3] in radians
     local ang = torch.uniform()*0.6 - 0.3
     images = image.rotate(images, ang)
     flow = rotateFlow(flow, ang)

     -- Add Random Scale
     local sc = math.floor(torch.uniform(1e-2, 15))
     sc = 30/(sc+15)

     local imagesIn = image.scale(images, '*'..sc)
     local flowIn = image.scale(flow, '*'..sc)*sc   -- Notice the scaling of flow here
     -- Add Random Noise to the images
     imagesIn = imagesIn:add(torch.rand(imagesIn:size()):mul(0.1):float())
      -- do random crop
     local iW = imagesIn:size(3)
     local iH = imagesIn:size(2)

     local oW = loadSize[3]
     local oH = loadSize[2]
     local h1 = math.floor(torch.uniform(1e-2, iH-oH))
     local w1 = math.floor(torch.uniform(1e-2, iW-oW))

     imagesOut = image.crop(imagesIn, w1, h1, w1 + oW, h1 + oH)
     flowOut = image.crop(flowIn, w1, h1, w1 + oW, h1 + oH)

     assert(imagesOut:size(3) == oW)
     assert(imagesOut:size(2) == oH)
     assert(flowOut:size(3) == oW)
     assert(flowOut:size(2) == oH)

     -- Augmentation and Contrast Normalization
     imagesOut = Preprocess()(imagesOut)
   else
    imagesOut = TF.ColorNormalize(meanstd)(images)
    flowOut = flow
   end

   return makeData(imagesOut, flowOut)
end

if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      loadSize = loadSize,
      inputSize = inputSize,
      outputSize = outputSize,
      split = 100,
      samplingIds = train_samples,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   trainLoader.sampleHookTrain = trainHook
end
collectgarbage()

-- End of train loader section
--------------------------------------------------------------------------------
--[[
   Section 2: Create a test data loader (testLoader),
   which can iterate over the test set and returns an image's
--]]

local testHook = function(self, id)
   collectgarbage()

   local path1 = paths.concat(opt.data, (string.format("%05i", id) .."_img1.ppm"))
   local path2 = paths.concat(opt.data, (string.format("%05i", id) .."_img2.ppm"))
   
   local img1 = loadImage(path1)
   local img2 = loadImage(path2)
   local images = torch.cat(img1, img2, 1)
   
   local pathF = paths.concat(opt.data, (string.format("%05i", id) .."_flow.flo"))
   local flow = flowX.loadFLO(pathF)

   images = TF.ColorNormalize(meanstd)(images)

   return makeData(images, flow)
end

if paths.filep(testCache) then
   print('Loading test metadata from cache')
   testLoader = torch.load(testCache)
   testLoader.sampleHookTest = testHook
else
   print('Creating test metadata')
   testLoader = dataLoader{
      loadSize = loadSize,
      inputSize = inputSize,
      outputSize = outputSize,
      split = 0,
      samplingIds = validation_samples,
      verbose = true
   }
   torch.save(testCache, testLoader)
   testLoader.sampleHookTest = testHook
end
collectgarbage()
-- End of test loader section
