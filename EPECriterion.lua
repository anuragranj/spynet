
-- Copyright 2016 Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.  
-- By using this software you agree to the terms of the license file 
-- in the root folder.
-- For commercial use, please contact ps-license@tue.mpg.de.

local EPECriterion, parent = torch.class('nn.EPECriterion', 'nn.Criterion')

-- Computes average endpoint error for batchSize x ChannelSize x Height x Width
-- flow fields or general multidimensional matrices.

local  eps = 1e-12

function EPECriterion:__init()
	parent.__init(self)
	self.sizeAverage = true
end

function EPECriterion:updateOutput(input, target)
	assert( input:nElement() == target:nElement(),
    "input and target size mismatch")

    self.buffer = self.buffer or input.new()

    local buffer = self.buffer
    local output
    local npixels

    buffer:resizeAs(input)
    npixels = input:nElement()/2    -- 2 channel flow fields

    buffer:add(input, -1, target):pow(2)
    output = torch.sum(buffer,2):sqrt()   -- second channel is flow
    output = output:sum()

    output = output / npixels

    self.output = output

    return self.output    
end

function EPECriterion:updateGradInput(input, target)

	assert( input:nElement() == target:nElement(),
    "input and target size mismatch")

    self.buffer = self.buffer or input.new()

    local buffer = self.buffer
    local gradInput = self.gradInput
    local npixels
    local loss

    buffer:resizeAs(input)
    npixels = input:nElement()/2

    buffer:add(input, -1, target):pow(2)
    loss = torch.sum(buffer,2):sqrt():add(eps)  -- forms the denominator
    loss = torch.cat(loss, loss, 2)   -- Repeat tensor to scale the gradients

    gradInput:resizeAs(input)
    gradInput:add(input, -1, target):cdiv(loss)
    gradInput = gradInput / npixels  
    return gradInput
end