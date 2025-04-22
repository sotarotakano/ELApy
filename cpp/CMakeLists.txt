cmake_minimum_required(VERSION 3.9)
project(pybindtest)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

add_subdirectory(carma)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Python 3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(Armadillo REQUIRED)

pybind11_add_module(StochOpt StochOpt.cpp)

target_compile_options(StochOpt PRIVATE -Wall -Wextra -pedantic -O3 -march=native)

target_include_directories(StochOpt PRIVATE
    ${ARMADILLO_INCLUDE_DIRS}
)

target_link_libraries(StochOpt PRIVATE
    pybind11::module
    ${ARMADILLO_LIBRARIES}
    ${BLAS_LIBRARIES}
    carma::carma
)


pybind11_add_module(ELA ELA.cpp)
target_compile_options(ELA PRIVATE -Wall -Wextra -pedantic -O3 -march=native)

target_include_directories(ELA PRIVATE 
    ${ARMADILLO_INCLUDE_DIRS}
)

target_link_libraries(ELA PRIVATE
    pybind11::module
    ${ARMADILLO_LIBRARIES}
    ${BLAS_LIBRARIES}
    carma::carma
)
