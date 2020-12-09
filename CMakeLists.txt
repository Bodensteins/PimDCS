cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pimtorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(pim_conv_mnist pim_conv_mnist.cpp logic_array_interface.h pim_linear.h pim_conv.h)
target_link_libraries(pim_conv_mnist "${TORCH_LIBRARIES}")
set_property(TARGET pim_conv_mnist PROPERTY CXX_STANDARD 14)

add_executable(pim_fc_mnist pim_fc_mnist.cpp logic_array_interface.h pim_linear.h)
target_link_libraries(pim_fc_mnist "${TORCH_LIBRARIES}")
set_property(TARGET pim_fc_mnist PROPERTY CXX_STANDARD 14)

add_executable(original_conv_mnist original_conv_mnist.cpp)
target_link_libraries(original_conv_mnist "${TORCH_LIBRARIES}")
set_property(TARGET original_conv_mnist PROPERTY CXX_STANDARD 14)

add_executable(original_fc_mnist original_fc_mnist.cpp)
target_link_libraries(original_fc_mnist "${TORCH_LIBRARIES}")
set_property(TARGET original_fc_mnist PROPERTY CXX_STANDARD 14)

# Test

add_executable(logic_array_interface_test logic_array_interface_test.cpp logic_array_interface.h)
target_link_libraries(logic_array_interface_test "${TORCH_LIBRARIES}")
set_property(TARGET logic_array_interface_test PROPERTY CXX_STANDARD 14)

add_executable(conv_test conv_test.cpp logic_array_interface.h pim_conv.h)
target_link_libraries(conv_test "${TORCH_LIBRARIES}")
set_property(TARGET conv_test PROPERTY CXX_STANDARD 14)