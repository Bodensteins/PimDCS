cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pimtorch)

#set(CMAKE_PREFIX_PATH "/etc/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS, "-std=c++14")
option(USE_OPENCV "Use opencv for object detection" OFF)


if (USE_OPENCV)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

include_directories(include)

if(APPLE)
    set(OPENMP_LIBRARIES "/usr/local/lib")
    set(OPENMP_INCLUDES "/usr/local/include")
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C "${CMAKE_C_COMPILER}")
        set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
        set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
        set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
        set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
        set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
        set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    endif()
endif()
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    if (APPLE)
        include_directories("${OPENMP_INCLUDES}")
        link_directories("${OPENMP_LIBRARIES}")
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Dependency
include(yaml-cpp.cmake)
include_directories(${YAML_CPP_INCLUDE_DIR})

# -DPIM_ARRAY_CONFIG_PATH=/your/path
if ("${PIM_ARRAY_CONFIG_PATH}" STREQUAL "")
    message(WARNING "Using default config path: ../config/pim_array_pro.yaml")
    set(PIM_ARRAY_CONFIG_PATH "../config/pim_array_pro.yaml")
else()
    message(STATUS "Using config path: ${PIM_ARRAY_CONFIG_PATH}")
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/config_path.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/config_path.h)

add_executable(pim_conv_mnist test/pim_conv_mnist.cpp)
add_dependencies(pim_conv_mnist YAML_CPP_EXTERNAL_PROJECT)
target_link_libraries(pim_conv_mnist "${TORCH_LIBRARIES}" "${YAML_CPP_LIBRARIES}")
set_property(TARGET pim_conv_mnist PROPERTY CXX_STANDARD 14)

add_executable(pim_fc_mnist test/pim_fc_mnist.cpp)
add_dependencies(pim_fc_mnist YAML_CPP_EXTERNAL_PROJECT)
target_link_libraries(pim_fc_mnist "${TORCH_LIBRARIES}" "${YAML_CPP_LIBRARIES}")
set_property(TARGET pim_fc_mnist PROPERTY CXX_STANDARD 14)

add_executable(original_conv_mnist test/original_conv_mnist.cpp)
add_dependencies(original_conv_mnist YAML_CPP_EXTERNAL_PROJECT)
target_link_libraries(original_conv_mnist "${TORCH_LIBRARIES}" "${YAML_CPP_LIBRARIES}")
set_property(TARGET original_conv_mnist PROPERTY CXX_STANDARD 14)

add_executable(original_fc_mnist test/original_fc_mnist.cpp)
add_dependencies(original_fc_mnist YAML_CPP_EXTERNAL_PROJECT)
target_link_libraries(original_fc_mnist "${TORCH_LIBRARIES}" "${YAML_CPP_LIBRARIES}")
set_property(TARGET original_fc_mnist PROPERTY CXX_STANDARD 14)

add_executable(original_vgg8_cifar10 test/original_vgg8_cifar10.cc)
add_dependencies(original_vgg8_cifar10 YAML_CPP_EXTERNAL_PROJECT)
target_link_libraries(original_vgg8_cifar10 "${TORCH_LIBRARIES}" "${YAML_CPP_LIBRARIES}")
set_property(TARGET original_vgg8_cifar10 PROPERTY CXX_STANDARD 14)

add_executable(pim_vgg8 test/pim_vgg8_cifar10.cc)
add_dependencies(pim_vgg8 YAML_CPP_EXTERNAL_PROJECT)
target_link_libraries(pim_vgg8 "${TORCH_LIBRARIES}" "${YAML_CPP_LIBRARIES}")
set_property(TARGET pim_vgg8 PROPERTY CXX_STANDARD 14)

add_executable(pim_fc_xor test/pim_fc_xor.cpp)
add_dependencies(pim_fc_xor YAML_CPP_EXTERNAL_PROJECT)
target_link_libraries(pim_fc_xor "${TORCH_LIBRARIES}" "${YAML_CPP_LIBRARIES}")
set_property(TARGET pim_fc_xor PROPERTY CXX_STANDARD 14)


# Test
add_executable(logic_array_interface_test unittest/logic_array_interface_test.cpp )
add_dependencies(logic_array_interface_test YAML_CPP_EXTERNAL_PROJECT)
target_link_libraries(logic_array_interface_test "${TORCH_LIBRARIES}" "${YAML_CPP_LIBRARIES}")
set_property(TARGET logic_array_interface_test PROPERTY CXX_STANDARD 14)

add_executable(conv_test test/conv_test.cpp)
add_dependencies(conv_test YAML_CPP_EXTERNAL_PROJECT)
target_link_libraries(conv_test "${TORCH_LIBRARIES}" "${YAML_CPP_LIBRARIES}")
set_property(TARGET conv_test PROPERTY CXX_STANDARD 14)

add_executable(linear_test test/linear_test.cpp)
add_dependencies(linear_test YAML_CPP_EXTERNAL_PROJECT)
target_link_libraries(linear_test "${TORCH_LIBRARIES}" "${YAML_CPP_LIBRARIES}")
set_property(TARGET linear_test PROPERTY CXX_STANDARD 14)

add_executable(simple_test unittest/simple_test.cpp)
add_dependencies(simple_test YAML_CPP_EXTERNAL_PROJECT)
target_link_libraries(simple_test "${TORCH_LIBRARIES}" "${YAML_CPP_LIBRARIES}")
set_property(TARGET simple_test PROPERTY CXX_STANDARD 14)

add_executable(unittestFor_pimPro unittest/unitTestFor_pimarraypro.cc)
add_dependencies(unittestFor_pimPro YAML_CPP_EXTERNAL_PROJECT)
target_link_libraries(unittestFor_pimPro "${TORCH_LIBRARIES}" "${YAML_CPP_LIBRARIES}")
set_property(TARGET unittestFor_pimPro PROPERTY CXX_STANDARD 14)


# cifar-10
add_executable(original_vgg test/vgg/original_vgg.cpp)
add_dependencies(original_vgg YAML_CPP_EXTERNAL_PROJECT)
target_link_libraries(original_vgg "${TORCH_LIBRARIES}" "${YAML_CPP_LIBRARIES}")
set_property(TARGET original_vgg PROPERTY CXX_STANDARD 14)

add_executable(pim_vgg test/vgg/pim_vgg.cpp)
add_dependencies(pim_vgg YAML_CPP_EXTERNAL_PROJECT)
target_link_libraries(pim_vgg "${TORCH_LIBRARIES}" "${YAML_CPP_LIBRARIES}")
set_property(TARGET pim_vgg PROPERTY CXX_STANDARD 14)

# yolo v3
if (USE_OPENCV)
    aux_source_directory(test/yolo_v3 YOLO_SRCS)
    add_executable(yolo_v3 ${YOLO_SRCS})
    add_dependencies(yolo_v3 YAML_CPP_EXTERNAL_PROJECT)
    target_link_libraries(yolo_v3 "${TORCH_LIBRARIES}" "${OpenCV_LIBS}" "${CURSES_LIBRARIES}" "${YAML_CPP_LIBRARIES}")
    set_property(TARGET yolo_v3 PROPERTY CXX_STANDARD 14)
endif ()


# online learning
add_subdirectory(test/online_learning)
