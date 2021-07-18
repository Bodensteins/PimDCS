project(online_learning)

find_library(JSONCPP_LIBRARIES NAMES jsoncpp HINTS /usr/local/lib REQUIRED)

add_subdirectory(visdom)
add_executable(training_onn training_onn.cpp progressbar.c)
add_dependencies(training_onn YAML_CPP_EXTERNAL_PROJECT)
target_link_libraries(training_onn ncurses visdom "${TORCH_LIBRARIES}" "${YAML_CPP_LIBRARIES}" "${JSONCPP_LIBRARIES}")
set_property(TARGET training_onn PROPERTY CXX_STANDARD 14)