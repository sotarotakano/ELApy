cmake_minimum_required(VERSION 3.16)
  
project(ELAtools)

find_package(Python COMPONENTS Interpreter Development)
set(CMAKE_CXX_STANDARD 14)
set(PYBIND11_CPP_STANDARD -std=c++14)

set(USE_ARMA_VERSION 14.0.x)
find_package(pybind11 CONFIG REQUIRED)
find_package(Armadillo REQUIRED)

add_subdirectory(carma)

pybind11_add_module(StochOpt SHARED StochOpt.cpp)
pybind11_add_module(ELA SHARED ELA.cpp)

# get system arch info
execute_process(
  COMMAND uname -m
  OUTPUT_VARIABLE SYSTEM_ARCHITECTURE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# get system arch ver info

message(STATUS "System architecture: ${SYSTEM_ARCHITECTURE}")

target_compile_options(StochOpt PRIVATE -Wall -Wextra -pedantic -O3 -march=native)
target_link_libraries(StochOpt PUBLIC pybind11::module)
target_link_libraries(StochOpt PRIVATE ${ARMADILLO_LIBRARIES} blas lapack)
target_include_directories(StochOpt PUBLIC ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(StochOpt PRIVATE carma::carma)

target_compile_options(ELA PRIVATE -Wall -Wextra -pedantic -O3 -march=native)
target_link_libraries(ELA PUBLIC pybind11::module)
target_link_libraries(ELA PRIVATE ${ARMADILLO_LIBRARIES} blas lapack)
target_include_directories(ELA PUBLIC ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(ELA PRIVATE carma::carma)
