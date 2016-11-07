# SPyNet: Spatial Pyramid Network for Optical Flow

## First things first
Install required packages

    cd extras/spybhwd
    luarocks make

    cd ../stnbhwd
    luarocks make

## The Fast way
Load an image pair

Download the data

    wget http://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/FlyingChairs.zip

Run the real-time test:

    th cudarealtimeTestedOk.lua
    
