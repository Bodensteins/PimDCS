cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pimtorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#include_directories(/usr/local/Cellar//python@3.8/3.8.5/Frameworks/Python.framework/Versions/3.8/include/python3.8)

# interface library
#add_library(logic_array_interface SHARED logic_array_interface.h)
#target_link_libraries(logic_array_interface "${TORCH_LIBRARIES}")
#set_property(TARGET logic_array_interface PROPERTY CXX_STANDARD 14)
#set_property(TARGET logic_array_interface PROPERTY LINKER_LANGUAGE CXX)
#target_compile_options(logic_array_interface PUBLIC -lpython -dynamiclib -shared -fPIC -Wl,-undefined,dynamic_lookup)
#
#add_library(simple_pim_linear SHARED pim_linear.h)
#target_link_libraries(simple_pim_linear "${TORCH_LIBRARIES}")
#set_property(TARGET simple_pim_linear PROPERTY CXX_STANDARD 14)
#set_property(TARGET simple_pim_linear PROPERTY LINKER_LANGUAGE CXX)
#target_compile_options(simple_pim_linear PUBLIC -shared -lpython -dynamiclib -fPIC -Wl,-undefined,dynamic_lookup)

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

add_executable(simple_test simple_test.cpp)
target_link_libraries(simple_test "${TORCH_LIBRARIES}")
set_property(TARGET simple_test PROPERTY CXX_STANDARD 14)