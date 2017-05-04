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
     A dataset class for loading images and dense outputs such as optical flow
     or segmentations in large datasets. Tested only on Linux (as it uses 
    command-line linux utilities to scale up)
]],
   {name="inputSize",
    type="table",
    help="the size of the input images"},

   {name="flowSize",
    type="table",
    help="the size of the network output"},

   {name="samplingMode",
    type="string",
    help="Sampling mode: random | balanced ",
    default = "balanced"},

   {name="verbose",
    type="boolean",
    help="Verbose mode during initialization",
    default = false},

   {name="samples",
    type="table",
    help="samples of training or testing images",
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

   if not self.sampleHookTrain then self.sampleHookTrain = self.defaultSampleHook end
   if not self.sampleHookTest then self.sampleHookTest = self.defaultSampleHook end

   local function tableFind(t, o) for k,v in pairs(t) do if v == o then return k end end end
  
   self.numSamples = #self.samples
   assert(self.numSamples > 0, "Could not find any sample in the given input paths")

   if self.verbose then print(self.numSamples ..  ' samples found.') end
end

-- size(), size(class)
function dataset:size(class, list)
  return self.numSamples
end


-- converts a table of samples (and corresponding labels) to a clean tensor
local function tableToOutput(self, imgTable, flowTable)
   local images, flows
   -- local quantity = #scalarTable
   local quantity = #imgTable
   --print('inputSize' .. self.inputSize[1])
   --print('image table size' .. imgTable[1]:size()[1])
   assert(imgTable[1]:size()[1] == self.inputSize[1])
   assert(flowTable[1]:size()[1] == self.flowSize[1])
   
   images = torch.Tensor(quantity,
           self.inputSize[1], self.inputSize[2], self.inputSize[3])
   flows = torch.Tensor(quantity,
           self.flowSize[1], self.flowSize[2], self.flowSize[3])

   for i=1,quantity do
      images[i]:copy(imgTable[i])
      flows[i]:copy(flowTable[i])
   end
   return images, flows
end

-- sampler, samples from the training set.
function dataset:sample(quantity)
   assert(quantity)
   local imgTable = {}
   local flowTable = {}
   --print('Quantity ' ..quantity)
   for i=1,quantity do
      local id = torch.random(1, self.numSamples)
      local img, flow = self:sampleHookTrain(id) -- single element[not tensor] from a row

      --print("Printing Image and Output Sizes in dataset sample")
      --print(img:size())
      --print(output:size())


      --local out = self:getById(id)
      table.insert(imgTable, img)
      table.insert(flowTable, flow)

   end
   -- print('Image table dim' .. imgTable[1]:dim() .. 'Output Table dim' .. outputTable[1]:dim()) 
   local images, flows = tableToOutput(self, imgTable, flowTable)
   return images, flows
end

function dataset:get(i1, i2)
   --local indices = self.samplingIds[{{i1, i2}}];
   local quantity = i2 - i1 + 1;
   assert(quantity > 0)
   -- now that indices has been initialized, get the samples
   local imgTable = {}
   local flowTable = {}

   for i=1,quantity do
      -- load the sample
      --print(indices[i], indices[i][1])
      local img, flow = self:sampleHookTest(i1+i-1)
      -- local out = self:sampleHookTest(imgpath)
      table.insert(imgTable, img)
      table.insert(flowTable, flow)
   end
   local images, flows = tableToOutput(self, imgTable, flowTable)
   return images, flows
end

return dataset
