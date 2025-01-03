cmake_minimum_required(VERSION 2.8)
project( DPM )
find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( dpm dpm_eval.cpp)
target_link_libraries( dpm ${OpenCV_LIBS})

