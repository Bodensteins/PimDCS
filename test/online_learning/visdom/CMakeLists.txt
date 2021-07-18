# Copyright (c) 2017-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
FIND_PACKAGE(CURL REQUIRED)
FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(Glog REQUIRED)
FIND_PACKAGE(Gflags REQUIRED)
FIND_PACKAGE(Threads)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

ADD_DEFINITIONS(-DCEREAL_RAPIDJSON_HAS_STDSTRING)

ADD_LIBRARY(visdom visdom.cpp)
SET_PROPERTY(TARGET visdom PROPERTY POSITION_INDEPENDENT_CODE ON)
message(${PROJECT_SOURCE_DIR})
TARGET_INCLUDE_DIRECTORIES(visdom
  PUBLIC
    "${CURL_INCLUDE_DIR}"
    "${GLOG_INCLUDE_DIRS}"
    "${GFLAGS_INCLUDE_DIRS}"
    # XXX Pick up rapidjson from cereal for now
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cereal/include/cereal/external/"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cereal/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty"
)

# Torch is here as a hack to find the right CUDA Paths. FindCUDA will work.
TARGET_LINK_LIBRARIES(visdom ${CURL_LIBRARIES} glog::glog ${GFLAGS_LIBRARIES} ${TORCH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
