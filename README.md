# spynet
Download the data

    wget http://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/FlyingChairs.zip

Install required packages:

    cd extras/spybhwd
    luarocks make

    cd extras/stnbhwd
    luarocks make

Run the real-time test:

    th cudarealtimeTestedOk.lua
    
