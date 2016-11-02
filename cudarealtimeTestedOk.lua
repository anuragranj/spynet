require 'image'
require 'cutorch'

--------------------------------
--LEVEL 5: LOSS IMAGE ON VALIDATION SET
--------------------------------
opt = {}
opt.showFlow = 0
opt.fineHeight = 384
opt.fineWidth = 512
opt.preprocess = 0
opt.level = 5 
opt.polluteFlow = 0
opt.augment = 0
opt.warp = 1
opt.batchSize = 1
opt.data = 'samples'
opt.N = 3
donkey = require('minidonkeyGPU')

local loss = torch.zeros(1,1, opt.fineHeight, opt.fineWidth):float()
local errors = torch.zeros(opt.N)
timings = torch.zeros(opt.N)
local loss = 0
local flowCPU = cutorch.createCudaHostTensor(640, 2,opt.fineHeight,opt.fineWidth):uniform()

for i=1,opt.N do
    collectgarbage()

    local id = i
    local imgs, flow = donkey.testHook(id)

    timer = torch.Timer()
    imgs = imgs:resize(1,6,opt.fineHeight, opt.fineWidth):cuda()
    flow_est = donkey.computeInitFlowL5(imgs):squeeze()
    flowCPU[i]:copyAsync(flow_est)
    cutorch.streamSynchronize(cutorch.getStream())
    local time_elapsed = timer:time().real  

    print('Time Elapsed: '..time_elapsed)

    timings[i] = time_elapsed
end
cutorch.streamSynchronize(cutorch.getStream())


for i=1,opt.N do
    local id = i
    local raw_im1, raw_im2, raw_flow = donkey.getRawData(id)
    
    
    local _err = (raw_flow - flowCPU[i]):pow(2)
    local err = torch.sum(_err, 1):sqrt()
    loss = loss + err:float()
    errors[i] = err:mean() 
    
    print(i, errors[i])
end
loss = torch.div(loss, opt.N)
print('Average EPE = '..loss:sum()/(opt.fineWidth*opt.fineHeight))
print('Mean Timing: ' ..timings:mean())
print('Median Timing: ' ..timings:median()[1])
