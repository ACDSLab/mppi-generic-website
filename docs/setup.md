---
title: "Setup"
permalink: /setup
description: "Installing or adding MPPI-Generic to your project"
layout: page
---

<!-- layout: page -->

# Installation of MPPI-Generic
On this page, we will try to give you a quick run-down on how to run MPPI-Generic on your own system.

## Prerequisites for installation
MPPI-Generic relies on the following:
* An NVIDIA GPU
* GCC/G++
* CUDA 10 or newer (CUDA 11.7+ is recommended but our library is compatible back to CUDA 10)
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
* [CMake](https://cmake.org/) 3.10 or newer
* git and git-lfs
### Unit tests requirements
* [yaml-cpp](https://github.com/jbeder/yaml-cpp)
* python-pil

### Prerequisite setup (Ubuntu)
1. Follow the instructions to install CUDA provided [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

2. Install all the other prerequisites through `apt-get`:
```bash
sudo apt-get install libeigen3-dev libyaml-cpp-dev git git-lfs cmake gcc
git lfs install
# extra installs if you are wanting to build unit tests
sudo apt-get install libyaml-cpp-dev python3-pil
```

## Using MPPI-Generic as a git submodule
As MPPI-Generic is still developing its API, our recommended way to incorporate this library into your projects
is through `git submodule`.
Git submodules allow you to keep track of specific commits of other projects so you can ensure that your code
will not break when the API is updated.
Setting up a `git submodule` is rather straightforward.
1. The following adds the MPPI-Generic codebase under a `submodules/MPPI-Generic` folder in your desired project.
```bash
cd /path/to/project-root
mkdir -p submodules
git submodule add https://github.gatech.edu/ACDS/MPPI-Generic.git submodules/MPPI-Generic
```
2. We now modify the root `CMakeLists.txt` of your project to add MPPI-Generic as a library your code can link to.
Note that this is a header-only library written in CUDA, so your executables/libraries will need to be `*.cu` files
for the correct compiler to be used.
First, we will modify the languages your CMake project uses to add C++/CUDA support.
```cmake
project(YOUR_PROJECT_NAME LANGUAGES CXX CUDA)
```
3. Next, we add our CMake module configuration that sets up compilation flags such as which NVIDIA GPU architecture
to compile for. After that, we add MPPI-Generic as a subdirectory and it is now ready for use.
```cmake
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/submodules/MPPI-Generic")
include(MPPIGenericToolsConfig)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/submodules/MPPI-Generic)
```
4. Adding MPPI-Generic to your target in CMake just requires linking to the `MPPI::MPPI` template library.
```cmake
add_executable(test_executable test.cu)
target_link_libraries(test_executable MPPI::MPPI)
```

## Installing MPPI-Generic
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
