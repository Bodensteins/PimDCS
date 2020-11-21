cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pimtorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(pimtorch main.cpp logic_array_interface.h)
target_link_libraries(pimtorch "${TORCH_LIBRARIES}")
set_property(TARGET pimtorch PROPERTY CXX_STANDARD 14)