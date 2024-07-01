layout: page
title: "Setup"
permalink: /setup

# Installation of MPPI-Generic
On this page, we will try to give you a quick run-down on how to run MPPI-Generic on your own system.

## Prerequisites for installation
MPPI-Generic relies on the following:
* An NVIDIA GPU
* CUDA 10 or newer [(Installation instructions)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
* [CMake](https://cmake.org/) 3.10 or newer
* git
### Prerequisite setup
Follow the instructions to install CUDA provided [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

Install all the other prerequisites through `apt-get`:
```bash
sudo apt-get install libeigen3-dev git cmake
```

## Get MPPI-Generic
If you want to use MPPI-Generic as a stand-alone library, we recommend cloning the git repo:
```bash
git clone mppi-generic-repo-url-here
cd MPPI-Generic
git submodule update --pull --recursive
```
You now have the MPPI-Generic library! [TODO: Check that make install works]
### Build examples
From the root directory of the MPPI-Generic repo:
```bash
mkdir build && cd build
cmake -DBUILD_EXAMPLES=ON .. # configure cmake to build examples
make
cd examples
```
In the `examples` folder, you will find multiple example programs using MPPI that should run
