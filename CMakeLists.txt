cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pimtorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 14)

add_executable(pim_conv_mnist test/pim_conv_mnist.cpp src/logic_array_interface.h src/pim_linear.h src/pim_conv.h)
target_link_libraries(pim_conv_mnist "${TORCH_LIBRARIES}")

add_executable(pim_fc_mnist test/pim_fc_mnist.cpp src/logic_array_interface.h src/pim_linear.h)
target_link_libraries(pim_fc_mnist "${TORCH_LIBRARIES}")

add_executable(original_conv_mnist test/original_conv_mnist.cpp)
target_link_libraries(original_conv_mnist "${TORCH_LIBRARIES}")

add_executable(original_fc_mnist test/original_fc_mnist.cpp)
target_link_libraries(original_fc_mnist "${TORCH_LIBRARIES}")

# cifar-10
add_executable(original_vgg vgg/original_vgg.cpp vgg/vgg.h)
target_link_libraries(original_vgg "${TORCH_LIBRARIES}")

add_executable(pim_vgg vgg/pim_vgg.cpp vgg/vgg.h)
target_link_libraries(pim_vgg "${TORCH_LIBRARIES}")

# Test
add_executable(logic_array_interface_test test/logic_array_interface_test.cpp src/logic_array_interface.h)
target_link_libraries(logic_array_interface_test "${TORCH_LIBRARIES}")

add_executable(conv_test test/conv_test.cpp src/logic_array_interface.h src/pim_conv.h)
target_link_libraries(conv_test "${TORCH_LIBRARIES}")

add_executable(simple_test test/simple_test.cpp)
target_link_libraries(simple_test "${TORCH_LIBRARIES}")

# yolo v3
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
aux_source_directory(yolo_v3 YOLO_SRCS)
add_executable(yolo_v3 ${YOLO_SRCS})
target_link_libraries(yolo_v3 "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")