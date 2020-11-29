cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pimtorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")



add_executable(mnist mnist.cpp logic_array_interface.h pim_linear.h)
target_link_libraries(mnist "${TORCH_LIBRARIES}")
set_property(TARGET mnist PROPERTY CXX_STANDARD 14)

add_executable(cnn_mnist mnist.cpp logic_array_interface.h pim_conv.h)
target_link_libraries(cnn_mnist "${TORCH_LIBRARIES}")
set_property(TARGET cnn_mnist PROPERTY CXX_STANDARD 14)

add_executable(original_mnist original_mnist.cpp logic_array_interface.h pim_linear.h)
target_link_libraries(original_mnist "${TORCH_LIBRARIES}")
set_property(TARGET original_mnist PROPERTY CXX_STANDARD 14)

add_executable(test test.cpp)
target_link_libraries(test "${TORCH_LIBRARIES}")
set_property(TARGET test PROPERTY CXX_STANDARD 14)

add_executable(logic_array_interface_test logic_array_interface_test.cpp logic_array_interface.h)
target_link_libraries(logic_array_interface_test "${TORCH_LIBRARIES}")
set_property(TARGET logic_array_interface_test PROPERTY CXX_STANDARD 14)