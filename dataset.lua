
-- Copyright 2016 Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.  
-- By using this software you agree to the terms of the license file 
-- in the root folder.
-- For commercial use, please contact ps-license@tue.mpg.de.

require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'

local dataset = torch.class('dataLoader')

local initcheck = argcheck{
   pack=true,
   help=[[
     A dataset class for images in a flat folder structure (folder-name is class-name).
     Optimized for extremely large datasets (upwards of 14 million images).
     Tested only on Linux (as it uses command-line linux utilities to scale up)
]],
   {name="inputSize",
    type="table",
    help="the size of the input images"},

   {name="outputSize",
    type="table",
    help="the size of the network output"},

   {name="split",
    type="number",
    help="Percentage of split to go to Training"
   },

   {name="samplingMode",
    type="string",
    help="Sampling mode: random | balanced ",
    default = "balanced"},

   {name="verbose",
    type="boolean",
    help="Verbose mode during initialization",
    default = false},

   {name="loadSize",
    type="table",
    help="a size to load the images to, initially",
    opt = true},

   {name="samplingIds",
    type="torch.LongTensor",
    help="the ids of training or testing images",
    opt = true},

   {name="sampleHookTrain",
    type="function",
    help="applied to sample during training(ex: for lighting jitter). "
       .. "It takes the image path as input",
    opt = true},

   {name="sampleHookTest",
    type="function",
    help="applied to sample during testing",
    opt = true},
}

function dataset:__init(...)

   -- argcheck
   local args =  initcheck(...)
   print(args)
   for k,v in pairs(args) do self[k] = v end

   if not self.loadSize then self.loadSize = self.inputSize; end

   if not self.sampleHookTrain then self.sampleHookTrain = self.defaultSampleHook end
   if not self.sampleHookTest then self.sampleHookTest = self.defaultSampleHook end

   local function tableFind(t, o) for k,v in pairs(t) do if v == o then return k end end end
   
   self.numSamples = self.samplingIds:size()[1]
   assert(self.numSamples > 0, "Could not find any sample in the given input paths")

   if self.verbose then print(self.numSamples ..  ' samples found.') end
end

function dataset:size(class, list)
  return self.numSamples
end

-- converts a table of samples (and corresponding labels) to a clean tensor
local function tableToOutput(self, imgTable, outputTable)
   local images, outputs
   local quantity = #imgTable
   assert(imgTable[1]:size()[1] == self.inputSize[1])
   assert(outputTable[1]:size()[1] == self.outputSize[1])
   
   images = torch.Tensor(quantity,
		       self.inputSize[1], self.inputSize[2], self.inputSize[3])
   outputs = torch.Tensor(quantity,
           self.outputSize[1], self.outputSize[2], self.outputSize[3])
   
   for i=1,quantity do
      images[i]:copy(imgTable[i])
      outputs[i]:copy(outputTable[i])
   end
   return images, outputs
end

-- sampler, samples from the training set.
function dataset:sample(quantity)
   assert(quantity)
   local imgTable = {}
   local outputTable = {}
   for i=1,quantity do
      local id = torch.random(1, self.numSamples)
      local img, output = self:sampleHookTrain(self.samplingIds[id][1]) -- single element[not tensor] from a row

      table.insert(imgTable, img)
      table.insert(outputTable, output)
   end
   local images, outputs = tableToOutput(self, imgTable, outputTable)
   return images, outputs
end

function dataset:get(i1, i2)
   local indices = self.samplingIds[{{i1, i2}}];
   local quantity = i2 - i1 + 1;
   assert(quantity > 0)
   local imgTable = {}
   local outputTable = {}
   for i=1,quantity do
      local img, output = self:sampleHookTest(indices[i][1])
      table.insert(imgTable, img)
      table.insert(outputTable, output)
   end
   local images, outputs = tableToOutput(self, imgTable, outputTable)
   return images, outputs
end

return dataset
