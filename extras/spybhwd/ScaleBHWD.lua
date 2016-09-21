local ScaleBHWD, parent = torch.class('nn.ScaleBHWD', 'nn.Module')

--[[
   ScaleBHWD() :
   ScaleBHWD:updateOutput({inputImages, grids})
   ScaleBHWD:updateGradInput({inputImages, grids}, gradOutput)

   ScaleBHWD will perform bilinear sampling of the input images according to the
   normalized coordinates provided in the grid. Output will be of same size as the grids, 
   with as many features as the input images.

   - inputImages has to be in BHWD layout

   - grids have to be in BHWD layout, with dim(D)=2
   - grids contains, for each sample (first dim), the normalized coordinates of the output wrt the input sample
      - first coordinate is Y coordinate, second is X
      - normalized coordinates : (-1,-1) points to top left, (-1,1) points to top right
      - if the normalized coordinates fall outside of the image, then output will be filled with zeros
]]

function ScaleBHWD:__init(scale)
   parent.__init(self)
   self.scale = scale or 1
end

function ScaleBHWD:check(input, gradOutput)
   local inputImages = input
--   local grids = input[2]

   assert(inputImages:isContiguous(), 'Input images have to be contiguous')
   assert(inputImages:nDimension()==4)
--   assert(grids:nDimension()==4)
--   assert(inputImages:size(1)==grids:size(1)) -- batch
--   assert(grids:size(4)==2) -- coordinates

--   if gradOutput then
-- TODO: checks for output size here
--      assert(inputImages:size(1)==gradOutput:size(1))
--      assert(inputImages:size(2)==gradOutput:size(2))
--      assert(inputImages:size(3)==gradOutput:size(3))
--   end
end

local function addOuterDim(t)
   local sizes = t:size()
   local newsizes = torch.LongStorage(sizes:size()+1)
   newsizes[1]=1
   for i=1,sizes:size() do
      newsizes[i+1]=sizes[i]
   end
   return t:view(newsizes)
end

function ScaleBHWD:updateOutput(input)
   local _inputImages = input
--   local _grids = input[2]

   local inputImages
   if _inputImages:nDimension()==3 then
      inputImages = addOuterDim(_inputImages)
--      grids = addOuterDim(_grids)
   else
      inputImages = _inputImages
--      grids = _grids
   end

   local input = inputImages

   self:check(input)

   self.output:resize(inputImages:size(1), self.scale*inputImages:size(2), self.scale*inputImages:size(3), inputImages:size(4))

   inputImages.nn.ScaleBHWD_updateOutput(self, inputImages, self.output)

   if _inputImages:nDimension()==3 then
      self.output=self.output:select(1,1)
   end
   
   return self.output
end

function ScaleBHWD:updateGradInput(_input, _gradOutput)
   self.gradInput:resizeAs(input)
   local _inputImages = _input

   local inputImages, gradOutput
   if _inputImages:nDimension()==3 then
      inputImages = addOuterDim(_inputImages)
      gradOutput = addOuterDim(_gradOutput)
   else
      inputImages = _inputImages
      gradOutput = _gradOutput
   end

   local input = inputImages

   self:check(input, gradOutput)
--   for i=1,#input do
   self.gradInput = self.gradInput or input.new()
   self.gradInput:resizeAs(input):zero()
--   end

   local gradInputImages = self.gradInput[1]
   --local gradGrids = self.gradInput[2]

   inputImages.nn.ScaleBHWD_updateGradInput(self, inputImages, gradInputImages, gradOutput)

   if _gradOutput:nDimension()==3 then
      self.gradInput=self.gradInput:select(1,1)
--      self.gradInput[2]=self.gradInput[2]:select(1,1)
   end
   
   return self.gradInput
end
