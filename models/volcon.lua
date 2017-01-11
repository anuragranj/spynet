-- Copyright 2016 Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.  
-- By using this software you agree to the terms of the license file 
-- in the root folder.
-- For commercial use, please contact ps-license@tue.mpg.de.

require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
function createModel(nGPU)
   local model = nn.Sequential()
   model:add(nn.SpatialConvolution(8,32,7,7,1,1,3,3))
   model:add(nn.ReLU())
   model:add(nn.SpatialConvolution(32,64,7,7,1,1,3,3))
   model:add(nn.ReLU())
   model:add(nn.SpatialConvolution(64,32,7,7,1,1,3,3))
   model:add(nn.ReLU())
   model:add(nn.SpatialConvolution(32,16,7,7,1,1,3,3))
   model:add(nn.ReLU())
   model:add(nn.SpatialConvolution(16,2,7,7,1,1,3,3))

   if nGPU>0 then
      model:cuda()
      model = makeDataParallel(model, nGPU)
   end
   
   return model
end
