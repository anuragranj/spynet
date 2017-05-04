-- Copyright 2016 Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.  
-- By using this software you agree to the terms of the license file 
-- in the root folder.
-- For commercial use, please contact ps-license@tue.mpg.de.

local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('SPyNet Coarse-to-Fine Optical Flow Training')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------

    cmd:option('-cache', 'checkpoint/', 'subdirectory in which to save/log experiments')
    cmd:option('-data', 'flying_chairs/data', 'Home of Flying Chairs dataset')
    cmd:option('-trainFile', 'trainDriveMonkaa.txt', 'Virtual Kitti Training Set')
    cmd:option('-valFile', 'val.txt', 'Virtual Kitti Validation Set')
    cmd:option('-manualSeed',         2, 'Manually set RNG seed')
    cmd:option('-GPU',                1, 'Default preferred GPU')
    cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
    cmd:option('-backend',     'cudnn', 'Options: cudnn | ccn2 | cunn')
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        4, 'number of donkeys to initialize (data loading threads)')
    cmd:option('-fineWidth',       512, 'the length of the fine flow field')
    cmd:option('-fineHeight',      384, 'the width of the fine flow field')
    cmd:option('-level',   1, 'Options: 1,2,3.., wheather to initialize flow to zero' )
    ------------- Training options --------------------
    cmd:option('-augment',         1,     'augment the data')   
    cmd:option('-nEpochs',         1000,    'Number of total epochs to run')
    cmd:option('-epochSize',       1000, 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       32,   'mini-batch size (1 = pure stochastic)')
    ---------- Optimization options ----------------------
    cmd:option('-LR',    0.0, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     5e-4, 'weight decay')
    cmd:option('-optimizer',       'adam', 'adam or sgd')
    ---------- Model options ----------------------------------
    cmd:option('-L1', 'models/modelL1_4.t7', 'Trained Level 1 model')
    cmd:option('-L2', 'models/modelL2_4.t7', 'Trained Level 2 model')
    cmd:option('-L3', 'models/modelL3_4.t7', 'Trained Level 3 model')
    cmd:option('-L4', 'models/modelL4_4.t7', 'Trained Level 4 model')

    cmd:option('-netType',     'volcon', 'Lua network file')
    cmd:option('-retrain',     'none', 'provide path to model to retrain with')
    cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
    cmd:text()

    local opt = cmd:parse(arg or {})
    opt.save = paths.concat(opt.cache)
    -- add date/time
    opt.save = paths.concat(opt.save, '' .. os.date():gsub(' ',''))
    
    opt.loadSize = {8, 384, 512} 
    return opt
end

return M
