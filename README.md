# ELApy
ELApy is the python package provides the interface of energy landscape analysis.

# Installation
ELApy is a python library that relies on the custom cpp modules based on the Rpackage([hiroakif93](https://github.com/hiroakif93)). The custom cpp modules are required to be built by CMake build with integrating "carma package" . See the following instruction for details.

It can be installed using:
```shell
echo "export pybind11_DIR=/Users/hoge/miniforge3/lib/python3.12/site-packages/pybind11" >> ~/.zshrc
echo “PATH=$PATH:$pybind11_DIR” >> ~/.zshrc
cd cpp
mkdir build && cd build
cmake ..
make
cd ..
find ./build -name”*.so” | xargs -I% mv % .
```

Reference
-----------
Suzuki, K., Nakaoka, S., Fukuda, S., & Masuya, H. (2021).
"Energy landscape analysis elucidates the multistability of ecological communities 
across environmental gradients." Ecological Monographs.
