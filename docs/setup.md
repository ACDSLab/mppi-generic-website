---
title: "Setup"
permalink: /setup
layout: page
---

<!-- layout: page -->

# Installation of MPPI-Generic
On this page, we will try to give you a quick run-down on how to run MPPI-Generic on your own system.

## Prerequisites for installation
MPPI-Generic relies on the following:
* An NVIDIA GPU
* GCC/G++
* CUDA 10 or newer [(Installation instructions)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
* [CMake](https://cmake.org/) 3.10 or newer
* git and git-lfs
* [yaml-cpp](https://github.com/jbeder/yaml-cpp) (for building unit tests)

### Prerequisite setup (Ubuntu)
1. Follow the instructions to install CUDA provided [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

2. Install all the other prerequisites through `apt-get`:
```bash
sudo apt-get install libeigen3-dev libyaml-cpp-dev git git-lfs cmake gcc
git lfs install
```

## Get MPPI-Generic
If you want to use MPPI-Generic as a stand-alone library, we recommend cloning the git repo:
```bash
git clone https://github.com/ACDSLab/MPPI-Generic.git
cd MPPI-Generic
git submodule update --init --recursive
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=~/.local ..
make && make install
```
You now have the MPPI-Generic library!

### Build examples or unit tests
From the root directory of the MPPI-Generic repo:
```bash
mkdir -p build && cd build
cmake -DBUILD_EXAMPLES=ON .. # configure cmake to build examples
make
cd examples
```
In the `examples` folder, you will find multiple example programs using MPPI that should run such as `cartpole_example`.
```bash
./cartpole_example
```
