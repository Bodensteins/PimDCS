cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pimtorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
add_executable(pim_conv_mnist pim_conv_mnist.cpp pure_array.h pim_linear.h pim_conv.h)
target_link_libraries(pim_conv_mnist "${TORCH_LIBRARIES}")
set_property(TARGET pim_conv_mnist PROPERTY CXX_STANDARD 14)

add_executable(pim_fc_mnist pim_fc_mnist.cpp pure_array.h pim_linear.h)
target_link_libraries(pim_fc_mnist "${TORCH_LIBRARIES}")
set_property(TARGET pim_fc_mnist PROPERTY CXX_STANDARD 14)

add_executable(original_conv_mnist original_conv_mnist.cpp)
target_link_libraries(original_conv_mnist "${TORCH_LIBRARIES}")
set_property(TARGET original_conv_mnist PROPERTY CXX_STANDARD 14)

add_executable(original_fc_mnist original_fc_mnist.cpp)
target_link_libraries(original_fc_mnist "${TORCH_LIBRARIES}")
set_property(TARGET original_fc_mnist PROPERTY CXX_STANDARD 14)

add_executable(unittest unitTestFor_pimArray.cc)
target_link_libraries(unittest "${TORCH_LIBRARIES}")
set_property(TARGET unittest PROPERTY CXX_STANDARD 14)
# Test

add_executable(logic_array_interface_test logic_array_interface_test.cpp pure_array.h)
target_link_libraries(logic_array_interface_test "${TORCH_LIBRARIES}")
set_property(TARGET logic_array_interface_test PROPERTY CXX_STANDARD 14)

add_executable(conv_test conv_test.cpp pure_array.h pim_conv.h)
target_link_libraries(conv_test "${TORCH_LIBRARIES}")
set_property(TARGET conv_test PROPERTY CXX_STANDARD 14)

add_executable(simple_test simple_test.cpp)
target_link_libraries(simple_test "${TORCH_LIBRARIES}")
set_property(TARGET simple_test PROPERTY CXX_STANDARD 14)