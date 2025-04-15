[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

# ELApy
ELApy provides the interface of energy landscape analysis ([kecosz](https://github.com/kecosz/ela)).

Reference
-----------
Suzuki, K., Nakaoka, S., Fukuda, S., & Masuya, H. (2021).
"Energy landscape analysis elucidates the multistability of ecological communities 
across environmental gradients." Ecological Monographs.

# Installation
ELApy is a python library that relies on the custom cpp modules based on the Rpackage([hiroakif93](https://github.com/hiroakif93)). 
The custom cpp modules are required to be built by CMake build with integrating "carma package" . See the following instruction for details.


### Step 1. Installing compiler and cpp packages by HomeBrew.

#### For Mac (arm64)
An installation of HomeBrew (if not installed)
```shell
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Compiler and cpp libraries are available on HomeBrew.
```shell
brew install gcc armadillo
```

#### For Linux
```shell
sudo apt -y install libtool texinfo libarmadillo-dev
```


### Step 2. Installing python libraries
ELApy is sensitive to the versions of numpy and pandas,
and the installation on virtual environment is strongly recommended.

(conda)
```shell
conda create -n ela python=3.12.9
```

```shell
conda activate ela
conda install -c conda-forge pybind11 pygam pandas numpy scipy=1.11.4
# numpy<=1.26.4, pandas==2.2.3, scipy<=1.11.4 is required for ELApy.
```

You can also install them via pip.
```shell
pip install pybind11 pygam numpy pandas 
```

```shell
# For visualiziation
conda install -c conda-forge seaborn matplotlib scikit-learn
```

### Step 3. Download ELApy and carma from GitHub

First download ELApy, then download carma (https://github.com/RUrlus/carma.) and place it in ELApy.

```shell
git clone "https://github.com/sotarotakano/ELApy.git"
cd ELApy
git clone "https://github.com/RUrlus/carma.git"
```

### Step 4. Building with CMake
The required cpp modules can be bullt using the attached CMakeLists.txt.
The built packages should be finally moved to ./cpp.

```shell
cd cpp
mkdir build && cd build
cmake ..
make
cd ..
find ./build -name”*.so” | xargs -I% mv % .
```
Although CMake basically can find pybind11 modules without additional settings,
the manual setting of environmenral path is usually required.
If the errors regarding pybind11 module appears during the built with CMake, please add "pybind11_DIR" to your $PATH.

(Following is an example. Please change it in such a way that it is compatible with your environment)
```shell
echo "export pybind11_DIR=/Users/hoge/miniforge3/lib/python3.12/site-packages/pybind11" >> ~/.zshrc
echo “PATH=$PATH:$pybind11_DIR” >> ~/.zshrc
```
Then let’s run
```shell
python test_run_ELA.py
```

If there is no error message, then everything is all set for you!

# Running with Google colab
If you want to try ELApy without the setup on your local environment, 
the easiest way is to run "ELApy_tutorial_colab.ipynb" using Google colab.
