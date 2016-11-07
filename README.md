# SPyNet: Spatial Pyramid Network for Optical Flow
This code is based on the paper [Optical Flow Estimation using a Spatial Pyramid Network](https://arxiv.org/abs/1611.00850)
## First things first
You need to have [Torch.](http://torch.ch/docs/getting-started.html#_)

Install other required packages
```bash
cd extras/spybhwd
luarocks make
cd ../stnbhwd
luarocks make
```
## For Fast Performace
#### Set up SPyNet
Set up SPyNet according to the image size and model. For optimal performance, resize your image such that width and height are a multiple of 32. You can also specify your favorite fine tuned model. The present supported modes are `sintelFinal`(default) and `sintelClean`. 
```lua
spynet = require('spynet')
computeFlow = spynet.setup(512, 384, 'sintelFinal')
```
Now you can call computeFlow anytime to estimate optical flow between image pairs.

#### Computing flow
Load an image pair and stack it.
```lua
require 'image'
im1 = image.load('samples/00001_img1.ppm' )
im2 = image.load('samples/00001_img2.ppm' )
im = torch.cat(im1, im2, 1)
```
SPyNet works with batches of data on CUDA. So, compute flow using
```lua
im = im:resize(1, im:size(1), im:size(2), im:size(3)):cuda()
flow = computeFlow(im)
```
You can also use batch-mode, if your images `im` are a tensor of size `Bx6xHxW`, of batch size B with 6 RGB pair channels. You can directly use:
```lua
flow = computeFlow(im)
```
## For Easy Usage
### Coming Soon.

## Timing Benchmarks
Our timing benchmark is set up on Flying chair dataset. To test it, you need to download
```bash
wget http://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/FlyingChairs.zip
```
Run the timing benchmark
```bash
th timing_benchmark.lua -data YOUR_FLYING_CHAIRS_DATA_DIRECTORY
```

## References
1. Our warping code is based on [qassemoquab/stnbhwd.](https://github.com/qassemoquab/stnbhwd)
2. The images in `samples` are from Flying Chairs dataset: 
   Dosovitskiy, Alexey, et al. "Flownet: Learning optical flow with convolutional networks." 2015 IEEE International Conference on Computer Vision (ICCV). IEEE, 2015.

## When using this code, please cite
Anurag Ranjan and Michael Black, "Optical Flow Estimation using a Spatial Pyramid Network", arXiv preprint arXiv:1611.00850 cs.CV (2016). 
