#!/bin/bash

# Enable debug mode
set -x

# TVM project directory
TVM_HOME=./3rdparty/tvm

# Navigate to TVM directory and print build message
echo "Building TVM"
cd $TVM_HOME

# If the argument is "rebuild", recreate the build folder
if [ "$1" == "rebuild" ]; then
    rm -rf build
fi

mkdir -p build
# Enter the build folder
cd build

if [ "$1" == "update" ]; then
    make -j4
else # normal build
    # Copy the config file and append necessary settings
    cp ../cmake/config.cmake .
    echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
    echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake
    echo "set(BUILD_STATIC_RUNTIME ON)" >> config.cmake
    echo "set(USE_MICRO ON)" >> config.cmake

    # Run cmake and make with 4 threads
    cmake .. && make -j4
fi

# Return to the root directory
cd - && cd ../..

# setup the tvm python package
export ABS_PATH=$(pwd)
export TVM_HOME=$ABS_PATH/3rdparty/tvm

python3 -m pip install -e $TVM_HOME/python

# validate the installation
python3 -c "import tvm; print(tvm.__file__)"
