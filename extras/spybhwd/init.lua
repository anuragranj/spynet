require 'nn'
local withCuda = pcall(require, 'cutorch')

require 'libspy'
if withCuda then
   require 'libcuspy'
end

require('spy.ScaleBHWD')

return nn
