# SPyNet: Spatial Pyramid Network for Optical Flow

## First things first
You need to have [Torch.](http://torch.ch/docs/getting-started.html#_)

Install other required packages

    cd extras/spybhwd
    luarocks make

    cd ../stnbhwd
    luarocks make

## For Fast Performace
#### Set up SPyNet
Set up SPyNet according to the image size and model. For optimal performance, make sure the image width and height are a multiple of 32. You can also specify your favorite fine tuned model. The present supported modes are `sintelFinal`(default) and `sintelClean`. 

    spynet = require('spynet')
    computeFlow = spynet.setup(512, 384, 'sintelFinal')

Now you can call computeFlow anytime to estimate optical flow between image pairs.

#### Computing flow
Load an image pair and stack it.

    require 'image'
    im1 = image.load('samples/00001_img1.ppm' )
    im2 = image.load('samples/00001_img2.ppm' )
    im = torch.cat(im1, im2, 1)

SPyNet works with batches of data on CUDA. So, compute flow using

    im = im:resize(1, im:size(1), im:size(2), im:size(3)):cuda()
    flow = computeFlow(im)

You can also use batch-mode, if your images `im` are a tensor of size `Bx6xHxW`, of batch size B with 6 RGB pair channels. You can directly use:

    flow = computeFlow(im)

## For Easy Usage
### Coming Soon.

## Timing Benchmarks
Our timing benchmark is set up on Flying chair dataset. To test it, you need to download

    wget http://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/FlyingChairs.zip

Run the timing benchmark

    th timing_benchmark.lua -data YOUR_FLYING_CHAIRS_DATA_DIRECTORY
    
