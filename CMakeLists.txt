cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pimtorch)
#SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
set(CMAKE_PREFIX_PATH /etc/libtorch/)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(include)
link_directories(lib)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
add_executable(pim_conv_mnist test/pim_conv_mnist.cpp)
target_link_libraries(pim_conv_mnist "${TORCH_LIBRARIES}" yaml-cpp)
set_property(TARGET pim_conv_mnist PROPERTY CXX_STANDARD 14)

add_executable(pim_fc_mnist test/pim_fc_mnist.cpp)
target_link_libraries(pim_fc_mnist "${TORCH_LIBRARIES}" yaml-cpp)
set_property(TARGET pim_fc_mnist PROPERTY CXX_STANDARD 14)

add_executable(pim_fc_onlycnt_mnist test/pim_fc_onlyCnt_mnist.cpp)
target_link_libraries(pim_fc_onlycnt_mnist "${TORCH_LIBRARIES}" yaml-cpp)
set_property(TARGET pim_fc_onlycnt_mnist PROPERTY CXX_STANDARD 14)

add_executable(original_conv_mnist test/original_conv_mnist.cpp)
target_link_libraries(original_conv_mnist "${TORCH_LIBRARIES}" yaml-cpp)
set_property(TARGET original_conv_mnist PROPERTY CXX_STANDARD 14)

add_executable(original_fc_mnist test/original_fc_mnist.cpp)
target_link_libraries(original_fc_mnist "${TORCH_LIBRARIES}" yaml-cpp)
set_property(TARGET original_fc_mnist PROPERTY CXX_STANDARD 14)

add_executable(unittest unittest/unitTestFor_pimArray.cc)
target_link_libraries(unittest "${TORCH_LIBRARIES}" yaml-cpp)
set_property(TARGET unittest PROPERTY CXX_STANDARD 14)

add_executable(unittest_only_counters unittest/unitTestFor_onlyCounters.cc)
target_link_libraries(unittest_only_counters "${TORCH_LIBRARIES}" yaml-cpp)
set_property(TARGET unittest_only_counters PROPERTY CXX_STANDARD 14)

add_executable(original_vgg8_cifar10 test/original_vgg8_cifar10.cc)
target_link_libraries(original_vgg8_cifar10 "${TORCH_LIBRARIES}" yaml-cpp)
set_property(TARGET original_vgg8_cifar10 PROPERTY CXX_STANDARD 14)

add_executable(pim_vgg8 test/pim_vgg8_cifar10.cc)
target_link_libraries(pim_vgg8 "${TORCH_LIBRARIES}" yaml-cpp)
set_property(TARGET pim_vgg8 PROPERTY CXX_STANDARD 14)


# Test


add_executable(logic_array_interface_test unittest/logic_array_interface_test.cpp )
target_link_libraries(logic_array_interface_test "${TORCH_LIBRARIES}" yaml-cpp)
set_property(TARGET logic_array_interface_test PROPERTY CXX_STANDARD 14)

add_executable(conv_test unittest/conv_test.cpp)
target_link_libraries(conv_test "${TORCH_LIBRARIES}" yaml-cpp)
set_property(TARGET conv_test PROPERTY CXX_STANDARD 14)

add_executable(simple_test unittest/simple_test.cpp)
target_link_libraries(simple_test "${TORCH_LIBRARIES}" yaml-cpp)
set_property(TARGET simple_test PROPERTY CXX_STANDARD 14)

add_executable(unittestFor_pimPro unittest/unitTestFor_pimarraypro.cc)
target_link_libraries(unittestFor_pimPro "${TORCH_LIBRARIES}" yaml-cpp)
set_property(TARGET unittestFor_pimPro PROPERTY CXX_STANDARD 14)


# cifar-10
add_executable(original_vgg test/vgg/original_vgg.cpp)
target_link_libraries(original_vgg "${TORCH_LIBRARIES}" yaml-cpp)

add_executable(pim_vgg test/vgg/pim_vgg.cpp)
target_link_libraries(pim_vgg "${TORCH_LIBRARIES}" yaml-cpp)


# # yolo v3
# aux_source_directory(yolo_v3 YOLO_SRCS)
# add_executable(yolo_v3 ${YOLO_SRCS})
# target_link_libraries(yolo_v3 "${TORCH_LIBRARIES}" "${OpenCV_LIBS}" "${YAML_CPP_LIBS}")
# include_directories(include)
# link_directories(lib)





