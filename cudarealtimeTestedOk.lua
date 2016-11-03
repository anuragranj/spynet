require 'image'
require 'cutorch'

--------------------------------
--LEVEL 5: LOSS IMAGE ON VALIDATION SET
--------------------------------
opt = {}
opt.showFlow = 0
opt.preprocess = 0
opt.level = 4 
opt.polluteFlow = 0
opt.augment = 0
opt.warp = 1
opt.batchSize = 1
opt.data = 'samples'
opt.N = 3
donkey = require('minidonkeyGPU')

local computeFlow
if opt.level == 5 then
    opt.fineHeight = 384
    opt.fineWidth = 512
    computeFlow = donkey.computeInitFlowL5
end

if opt.level == 4 then
    opt.fineHeight = 192
    opt.fineWidth = 256
    computeFlow = donkey.computeInitFlowL4
end

for i=1,opt.N do
    collectgarbage()

    local id = i
    local imgs, flow = donkey.testHook(id)
    imgs = image.scale(imgs, opt.fineWidth, opt.fineHeight)

    timer = torch.Timer()
    imgs = imgs:resize(1,6,opt.fineHeight, opt.fineWidth):cuda()
    flow_est = computeFlow(imgs):squeeze()
    cutorch.synchronize()
    local time_elapsed = timer:time().real  

    print('Time Elapsed: '..time_elapsed)
end