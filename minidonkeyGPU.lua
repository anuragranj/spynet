--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'image'
local TF = require 'transforms'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'stn'
require 'spy'

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

if opt.augment == 1 then
   modelL1path = paths.concat('models', 'modelL1_3.t7')
   modelL2path = paths.concat('models', 'modelL2_3.t7')
   modelL3path = paths.concat('models', 'modelL3_3.t7')
   modelL4path = paths.concat('models', 'modelL4_3.t7')
   modelL5path = paths.concat('models', 'modelL5_3.t7')

else
   modelL1path = paths.concat('models', 'modelL1_4.t7')
   modelL2path = paths.concat('models', 'modelL2_4.t7')
   modelL3path = paths.concat('models', 'modelL3_4.t7')
   modelL4path = paths.concat('models', 'modelL4_4.t7')
   modelL5path = paths.concat('models', 'modelL5_4.t7')
end


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
   --local numSamples = 512
   --print("WARNING: Using only " ..numSamples .." data points")
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
   -- find the smaller dimension, and resize it to loadSize (while keeping aspect ratio)
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
  --print('Image L1 size')
  --print(imagesL1:size())
  local h = 24
  local w = 32

--  local _flowappend = torch.CudaTensor(opt.batchSize, 2, h, w)
  local _flowappend = torch.zeros(opt.batchSize, 2, h, w):cuda()
  local images_in = torch.cat(imagesL1, _flowappend, 2)

  local flow_est = modelL1:forward(images_in)
  return flow_est
  -- body
end
M.computeInitFlowL1 = computeInitFlowL1

local function computeInitFlowL2(imagesL2)
--  local h = 48
--  local w = 64
  --print('Image L2 size')
  --print(imagesL2:size())
  --print(imagesL2:type())

  local imagesL1 = down2:forward(imagesL2:clone())
  --print('Image L2 size one')
  --print(imagesL2:size())
  
  local _flowappend = up2:forward(computeInitFlowL1(imagesL1))*2  
--  print('Image L2 size two')
--  print(imagesL2:size())
  
  local _img2 = imagesL2[{{},{4,6},{},{}}]
--  print('Image L2 size three')
--  print(imagesL2:size())

--  print('warp img size and _img2 size')
--  print(warpImg:size())
--  print(_img2:size())
--  imagesL2[{{},{4,6},{},{}}]:copy(warpmodel2:forward({_img2, _flowappend:index(2, torch.LongTensor{2,1})}))
  imagesL2[{{},{4,6},{},{}}]:copy(warpmodel2:forward({_img2, _flowappend}))

  local images_in = torch.cat(imagesL2, _flowappend, 2)
  
  local  flow_est = modelL2:forward(images_in)
  return flow_est:add(_flowappend)
  -- body
end
M.computeInitFlowL2 = computeInitFlowL2

local function computeInitFlowL3(imagesL3)
  --print('Image L3 size')
  --print(imagesL3:size())

  local imagesL2 = down3:forward(imagesL3:clone())
  local _flowappend = up3:forward(computeInitFlowL2(imagesL2))*2  
  local _img2 = imagesL3[{{},{4,6},{},{}}]
--  imagesL3[{{},{4,6},{},{}}]:copy(warpmodel3:forward({_img2, _flowappend:index(2, torch.LongTensor{2,1})}))
  imagesL3[{{},{4,6},{},{}}]:copy(warpmodel3:forward({_img2, _flowappend}))

  local images_in = torch.cat(imagesL3, _flowappend, 2)
  
  local  flow_est = modelL3:forward(images_in)
  return flow_est:add(_flowappend)
  -- body
end
M.computeInitFlowL3 = computeInitFlowL3

local  function computeInitFlowL4(imagesL4)
  --print('Image L4 size')
  --print(imagesL4:size())

  local imagesL3 = down4:forward(imagesL4)
  local _flowappend = up4:forward(computeInitFlowL3(imagesL3))*2  
  local _img2 = imagesL4[{{},{4,6},{},{}}]
--  imagesL4[{{},{4,6},{},{}}]:copy(warpmodel4:forward({_img2, _flowappend:index(2, torch.LongTensor{2,1})}))
  imagesL4[{{},{4,6},{},{}}]:copy(warpmodel4:forward({_img2, _flowappend}))

  local images_in = torch.cat(imagesL4, _flowappend, 2)
  
  local  flow_est = modelL4:forward(images_in)
  return flow_est:add(_flowappend)  -- body
end
M.computeInitFlowL4 = computeInitFlowL4

local function computeInitFlowL5(imagesL5)
  --print('Image L5 size')
  --print(imagesL5:size())
  
  local imagesL4 = down5:forward(imagesL5)
  local _flowappend = up5:forward(computeInitFlowL4(imagesL4))*2  
  
  local _img2 = imagesL5[{{},{4,6},{},{}}]
--  imagesL5[{{},{4,6},{},{}}]:copy(warpmodel5:forward({_img2, _flowappend:index(2, torch.LongTensor{2,1})}))
  imagesL5[{{},{4,6},{},{}}]:copy(warpmodel5:forward({_img2, _flowappend}))

  local images_in = torch.cat(imagesL5, _flowappend, 2)
  
  local  flow_est = modelL5:forward(images_in)
  return flow_est:add(_flowappend)  -- body
  -- body
end
M.computeInitFlowL5 = computeInitFlowL5

local function makeData(images, flows)
  images = images:cuda()
  local imagesL4 = down:forward(images)
  local imagesL3 = down:forward(imagesL4)
  local imagesL2 = down:forward(imagesL3)
  local imagesL1 = down:forward(imagesL2)

  local initFlow, flowDiffOutput
  local images_scaled = image.scale(images, opt.fineWidth, opt.fineHeight)
  
  if opt.level == 0 then
    initFlow = torch.zeros(2, opt.fineHeight, opt.fineWidth)
    flowDiffOutput = scaleFlow(flows, opt.fineHeight, opt.fineWidth) 

  elseif opt.level == 1 then
    initFlow = torch.zeros(2, opt.fineHeight, opt.fineWidth)
    flowDiffOutput = scaleFlow(flows, opt.fineHeight, opt.fineWidth)

  elseif opt.level == 2 then
    assert(opt.fineWidth==64, 'Level width mismatch')
    initFlow = computeInitFlowL1(imagesL1)
    initFlow = scaleFlow(initFlow, opt.fineHeight, opt.fineWidth)

    flowDiffOutput = scaleFlow(flows, opt.fineHeight, opt.fineWidth)
    flowDiffOutput = flowDiffOutput:add(flowDiffOutput, -1, initFlow)

  elseif opt.level == 3 then
    assert(opt.fineWidth==128, 'Level width mismatch')
    initFlow = computeInitFlowL2(imagesL2)
    initFlow = scaleFlow(initFlow, opt.fineHeight, opt.fineWidth)

    flowDiffOutput = scaleFlow(flows, opt.fineHeight, opt.fineWidth)
    flowDiffOutput = flowDiffOutput:add(flowDiffOutput, -1, initFlow)

  elseif opt.level == 4 then
    assert(opt.fineWidth == 256, 'Level width mismatch')
    initFlow = computeInitFlowL3(imagesL3)
    initFlow = scaleFlow(initFlow, opt.fineHeight, opt.fineWidth)

    flowDiffOutput = scaleFlow(flows, opt.fineHeight, opt.fineWidth)
    flowDiffOutput = flowDiffOutput:add(flowDiffOutput, -1, initFlow)

  elseif opt.level == 5 then
    assert(opt.fineWidth == 512, 'Level width mismatch')
    initFlow = computeInitFlowL4(imagesL4)
    initFlow = scaleFlow(initFlow, opt.fineHeight, opt.fineWidth)
    
    flowDiffOutput = scaleFlow(flows, opt.fineHeight, opt.fineWidth)
    flowDiffOutput = flowDiffOutput:add(flowDiffOutput, -1, initFlow)

  end

  if opt.warp == 1 then
    local _img2 = images_scaled[{{4,6},{},{}}]:clone()
    images_scaled[{{4,6},{},{}}]:copy(image.warp(_img2, initFlow:index(1, torch.LongTensor{2,1})))
  end

  if opt.polluteFlow == 1 then
    initFlow = initFlow + torch.rand(initFlow:size()):mul(2):csub(1)
  end

  local imageFlowInputs = torch.cat(images_scaled, initFlow:float(), 1)

  --print('Printing makeData')
  --print(imageFlowInputs:size())
  --print(flowDiffOutput:size())

  return imageFlowInputs, flowDiffOutput
end

M.makeData = makeData

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
   
--   for i=1,6 do -- channels
--      if mean then images[{{i},{},{}}]:add(-mean[i]) end
--      if std then images[{{i},{},{}}]:div(std[i]) end
--   end
   images = TF.ColorNormalize(meanstd)(images)


   return images, flow
end
M.testHook = testHook

local function writeFlow(filename, F)
  F = F:permute(2,3,1):clone()
  TAG_FLOAT = 202021.25 
  local ff = torch.DiskFile(filename, 'w'):binary()
  ff:writeFloat(TAG_FLOAT)
   
  ff:writeInt(F:size(2)) -- width
  ff:writeInt(F:size(1)) -- height

  ff:writeFloat(F:storage())
  ff:close()
end
M.writeFlow = writeFlow

local saveHook = function(id, flow, saveDir)
   local pathF = paths.concat(saveDir, (string.format("%05i", id) .."_flow.flo"))
   print('Saving to ' ..pathF)
   writeFlow(pathF, flow)

end
M.saveHook = saveHook

---------------
-- FLOW UTILS
---------------
local function computeNorm(...)
   -- check args
   local _, flow_x, flow_y = xlua.unpack(
      {...},
      'opticalflow.computeNorm',
      'computes norm (size) of flow field from flow_x and flow_y,\n',
      {arg='flow_x', type='torch.Tensor', help='flow field (x), (WxH)', req=true},
      {arg='flow_y', type='torch.Tensor', help='flow field (y), (WxH)', req=true}
   )
   local flow_norm = torch.Tensor()
   local x_squared = torch.Tensor():resizeAs(flow_x):copy(flow_x):cmul(flow_x)
   flow_norm:resizeAs(flow_y):copy(flow_y):cmul(flow_y):add(x_squared):sqrt()
   return flow_norm
end
M.computeNorm = computeNorm
------------------------------------------------------------
-- computes angle (direction) of flow field from flow_x and flow_y,
--
-- @usage opticalflow.computeAngle() -- prints online help
--
-- @param flow_x  flow field (x), (WxH) [required] [type = torch.Tensor]
-- @param flow_y  flow field (y), (WxH) [required] [type = torch.Tensor]
------------------------------------------------------------
local function computeAngle(...)
   -- check args
   local _, flow_x, flow_y = xlua.unpack(
      {...},
      'opticalflow.computeAngle',
      'computes angle (direction) of flow field from flow_x and flow_y,\n',
      {arg='flow_x', type='torch.Tensor', help='flow field (x), (WxH)', req=true},
      {arg='flow_y', type='torch.Tensor', help='flow field (y), (WxH)', req=true}
   )
   local flow_angle = torch.Tensor()
   flow_angle:resizeAs(flow_y):copy(flow_y):cdiv(flow_x):abs():atan():mul(180/math.pi)
   flow_angle:map2(flow_x, flow_y, function(h,x,y)
				      if x == 0 and y >= 0 then
					 return 90
				      elseif x == 0 and y <= 0 then
					 return 270
				      elseif x >= 0 and y >= 0 then
					 -- all good
				      elseif x >= 0 and y < 0 then
					 return 360 - h
				      elseif x < 0 and y >= 0 then
					 return 180 - h
				      elseif x < 0 and y < 0 then
					 return 180 + h
				      end
				   end)
   return flow_angle
end
M.computeAngle = computeAngle
------------------------------------------------------------
-- merges Norm and Angle flow fields into a single RGB image,
-- where saturation=intensity, and hue=direction
--
-- @usage opticalflow.field2rgb() -- prints online help
--
-- @param norm  flow field (norm), (WxH) [required] [type = torch.Tensor]
-- @param angle  flow field (angle), (WxH) [required] [type = torch.Tensor]
-- @param max  if not provided, norm:max() is used [type = number]
-- @param legend  prints a legend on the image [type = boolean]
------------------------------------------------------------
local function field2rgb(...)
   -- check args
   local _, norm, angle, max, legend = xlua.unpack(
      {...},
      'opticalflow.field2rgb',
      'merges Norm and Angle flow fields into a single RGB image,\n'
	 .. 'where saturation=intensity, and hue=direction',
      {arg='norm', type='torch.Tensor', help='flow field (norm), (WxH)', req=true},
      {arg='angle', type='torch.Tensor', help='flow field (angle), (WxH)', req=true},
      {arg='max', type='number', help='if not provided, norm:max() is used'},
      {arg='legend', type='boolean', help='prints a legend on the image', default=false}
   )
   
   -- max
   local saturate = false
   if max then saturate = true end
   max = math.max(max or norm:max(), 1e-2)
   
   -- merge them into an HSL image
   local hsl = torch.Tensor(3,norm:size(1), norm:size(2))
   -- hue = angle:
   hsl:select(1,1):copy(angle):div(360)
   -- saturation = normalized intensity:
   hsl:select(1,2):copy(norm):div(max)
   if saturate then hsl:select(1,2):tanh() end
   -- light varies inversely from saturation (null flow = white):
   hsl:select(1,3):copy(hsl:select(1,2)):mul(-0.5):add(1)
   
   -- convert HSL to RGB
   local rgb = image.hsl2rgb(hsl)
   
   -- legend
   if legend then
      _legend_ = _legend_
	 or image.load(paths.concat(paths.install_lua_path, 'opticalflow/legend.png'),3)
      legend = torch.Tensor(3,hsl:size(2)/8, hsl:size(2)/8)
      image.scale(_legend_, legend, 'bilinear')
      rgb:narrow(1,1,legend:size(2)):narrow(2,hsl:size(2)-legend:size(2)+1,legend:size(2)):copy(legend)
   end
   
   -- done
   return rgb
end
M.field2rgb = field2rgb
------------------------------------------------------------
-- Simplifies display of flow field in HSV colorspace when the
-- available field is in x,y displacement
--
-- @usage opticalflow.xy2rgb() -- prints online help
--
-- @param x  flow field (x), (WxH) [required] [type = torch.Tensor]
-- @param y  flow field (y), (WxH) [required] [type = torch.Tensor]
------------------------------------------------------------
local function xy2rgb(...)
   -- check args
   local _, x, y, max = xlua.unpack(
      {...},
      'opticalflow.xy2rgb',
      'merges x and y flow fields into a single RGB image,\n'
	 .. 'where saturation=intensity, and hue=direction',
      {arg='x', type='torch.Tensor', help='flow field (norm), (WxH)', req=true},
      {arg='y', type='torch.Tensor', help='flow field (angle), (WxH)', req=true},
      {arg='max', type='number', help='if not provided, norm:max() is used'}
   )
   
   local norm = computeNorm(x,y)
   local angle = computeAngle(x,y)
   return field2rgb(norm,angle,max)
end
M.xy2rgb = xy2rgb

-----------------
-----------------
return M


