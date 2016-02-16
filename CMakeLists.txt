cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)

set(opencv_3_1_path add_path_to_your_opencv_specific_install_if_needed)

project( DisplayImage )
find_package( OpenCV REQUIRED NO_DEFAULT_PATH HINTS ${opencv_3_1_path} )
#find_package( OpenCV REQUIRED)
add_executable( TestSolvePnp TestSolvePnp.cpp epnp/epnp.cpp)
target_link_libraries( TestSolvePnp ${OpenCV_LIBS} )
