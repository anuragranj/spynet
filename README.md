# SPyNet: Spatial Pyramid Network for Optical Flow
This code is based on the paper [Optical Flow Estimation using a Spatial Pyramid Network](https://arxiv.org/abs/1611.00850). Skip to:

* [First things first:](#setUp)  Setting up this code
* [Easy Usage:](#easyUsage) Compute Optical Flow in 5 lines
* [Fast Performance Usage:](#fastPerformanceUsage) Compute Optical Flow at a rocket speed
* [Training:](#training) Train your own models using Spatial Pyramid approach on mulitiple GPUs.
* [Optical Flow Utilities:](#flowUtils) A set of functions in lua for working around optical flow
* [References:](#references) For further reading

<a name="setUp"></a>
## First things first
You need to have [Torch.](http://torch.ch/docs/getting-started.html#_)

Install other required packages
```bash
cd extras/spybhwd
luarocks make
cd ../stnbhwd
luarocks make
```
<a name="easyUsage"></a>
## For Easy Usage, follow this
#### Set up SPyNet
```lua
spynet = require('spynet')
easyComputeFlow = spynet.easy_setup()
```
#### Load images and compute flow
```lua
im1 = image.load('samples/00001_img1.ppm' )
im2 = image.load('samples/00001_img2.ppm' )
flow = easyComputeFlow(im1, im2)
```
<a name="fastPerformanceUsage"></a>
## For Fast Performace, follow this (recommended)
#### Set up SPyNet
Set up SPyNet according to the image size and model. For optimal performance, resize your image such that width and height are a multiple of 32. You can also specify your favorite model. The present supported modes are fine tuned models `sintelFinal`(default), `sintelClean`, `kittiFinal`, and base models `chairsFinal` and `chairsClean`. 
```lua
spynet = require('spynet')
computeFlow = spynet.setup(512, 384, 'sintelFinal')    -- for 384x512 images
```
Now you can call computeFlow anytime to estimate optical flow between image pairs.

#### Computing flow
Load an image pair and stack and normalize it.
```lua
im1 = image.load('samples/00001_img1.ppm' )
im2 = image.load('samples/00001_img2.ppm' )
im = torch.cat(im1, im2, 1)
im = spynet.normalize(im)
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
<a name="training"></a>
## Training
Training sequentially is faster than training end-to-end since you need to learn small number of parameters at each level. To train a level `N`, we need the trained models at levels `1` to `N-1`. You also initialize the model with a pretrained model at `N-1`.

E.g. To train level 3, we need trained models at `L1` and `L2`, and we initialize it  `modelL2_3.t7`.
```bash
th main.lua -fineWidth 128 -fineHeight 96 -level 3 -netType volcon \
-cache checkpoint -data FLYING_CHAIRS_DIR \
-L1 models/modelL1_3.t7 -L2 models/modelL2_3.t7 \
-retrain models/modelL2_3.t7
```
<a name="flowUtils"></a>
## Optical Flow Utilities
We provide `flowExtensions.lua` containing various functions to make your life easier with optical flow while using Torch/Lua. You can just copy this file into your project directory and use if off the shelf.
```lua
flowX = require 'flowExtensions'
```
#### [flow_magnitude] flowX.computeNorm(flow_x, flow_y)
Given `flow_x` and `flow_y` of size `MxN` each, evaluate `flow_magnitude` of size `MxN`.

#### [flow_angle] flowX.computeAngle(flow_x, flow_y)
Given `flow_x` and `flow_y` of size `MxN` each, evaluate `flow_angle` of size `MxN` in degrees.

#### [rgb] flowX.field2rgb(flow_magnitude, flow_angle, [max], [legend])
Given `flow_magnitude` and `flow_angle` of size `MxN` each, return an image of size `3xMxN` for visualizing optical flow. `max`(optional) specifies maximum flow magnitude and `legend`(optional) is boolean that prints a legend on the image.

#### [rgb] flowX.xy2rgb(flow_x, flow_y, [max])
Given `flow_x` and `flow_y` of size `MxN` each, return an image of size `3xMxN` for visualizing optical flow. `max`(optional) specifies maximum flow magnitude.

#### [flow] flowX.loadFLO(filename)
Reads a `.flo` file. Loads `x` and `y` components of optical flow in a 2 channel `2xMxN` optical flow field. First channel stores `x` component and second channel stores `y` component.

#### flowX.writeFLO(filename,F)
Write a `2xMxN` flow field `F` containing `x` and `y` components of its flow fields in its first and second channel respectively to `filename`, a `.flo` file.

#### [flow] flowX.loadPFM(filename)
Reads a `.pfm` file. Loads `x` and `y` components of optical flow in a 2 channel `2xMxN` optical flow field. First channel stores `x` component and second channel stores `y` component.

#### [flow_rotated] flowX.rotate(flow, angle)
Rotates `flow` of size `2xMxN` by `angle` in radians. Uses nearest-neighbor interpolation to avoid blurring at boundaries.

#### [flow_scaled] flowX.scale(flow, sc, [opt])
Scales `flow` of size `2xMxN` by `sc` times. `opt`(optional) specifies interpolation method, `simple` (default), `bilinear`, and `bicubic`.

#### [flowBatch_scaled] flowX.scaleBatch(flowBatch, sc)
Scales `flowBatch` of size `Bx2xMxN`, a batch of `B` flow fields by `sc` times. Uses nearest-neighbor interpolation.

<a name="timing"></a>
## Timing Benchmarks
Our timing benchmark is set up on Flying chair dataset. To test it, you need to download
```bash
wget http://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/FlyingChairs.zip
```
Run the timing benchmark
```bash
th timing_benchmark.lua -data YOUR_FLYING_CHAIRS_DATA_DIRECTORY
```

<a name="references"></a>
## References
1. Our warping code is based on [qassemoquab/stnbhwd.](https://github.com/qassemoquab/stnbhwd)
2. The images in `samples` are from Flying Chairs dataset: 
   Dosovitskiy, Alexey, et al. "Flownet: Learning optical flow with convolutional networks." 2015 IEEE International Conference on Computer Vision (ICCV). IEEE, 2015.
3. Some parts of `flowExtensions.lua` are adapted from [marcoscoffier/optical-flow](https://github.com/marcoscoffier/optical-flow/blob/master/init.lua) with help from [fguney](https://github.com/fguney).
   
## License
Free for non-commercial and scientific research purposes. For commercial use, please contact ps-license@tue.mpg.de. Check LICENSE file for details.

## When using this code, please cite
Ranjan, Anurag, and Michael J. Black. "Optical Flow Estimation using a Spatial Pyramid Network." arXiv preprint arXiv:1611.00850 (2016). 
