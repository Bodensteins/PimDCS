cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pimtorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 14)

add_executable(pim_conv_mnist pim_conv_mnist.cpp logic_array_interface.h pim_linear.h pim_conv.h)
target_link_libraries(pim_conv_mnist "${TORCH_LIBRARIES}")

add_executable(pim_fc_mnist pim_fc_mnist.cpp logic_array_interface.h pim_linear.h)
target_link_libraries(pim_fc_mnist "${TORCH_LIBRARIES}")

add_executable(original_conv_mnist original_conv_mnist.cpp)
target_link_libraries(original_conv_mnist "${TORCH_LIBRARIES}")

add_executable(original_fc_mnist original_fc_mnist.cpp)
target_link_libraries(original_fc_mnist "${TORCH_LIBRARIES}")

# cifar-10
add_executable(original_vgg original_vgg.cpp)
target_link_libraries(original_vgg "${TORCH_LIBRARIES}")

#add_executable(pim_vgg pim_vgg.cpp)
#target_link_libraries(pim_vgg "${TORCH_LIBRARIES}")

# Test
add_executable(logic_array_interface_test logic_array_interface_test.cpp logic_array_interface.h)
target_link_libraries(logic_array_interface_test "${TORCH_LIBRARIES}")

add_executable(conv_test conv_test.cpp logic_array_interface.h pim_conv.h)
target_link_libraries(conv_test "${TORCH_LIBRARIES}")

add_executable(simple_test simple_test.cpp)
target_link_libraries(simple_test "${TORCH_LIBRARIES}")