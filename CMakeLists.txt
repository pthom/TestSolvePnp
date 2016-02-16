cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)

#set(opencv_3_1_path /Users/pthomet@IVSDOMAIN.FR/dvp/_EyeCube/Z_GRepo/srcs/activisu/activisumodules-gitproxy/Externals/Opencv/opencv-itseezgit.install/usr/local/share/OpenCV)

project( DisplayImage )
find_package( OpenCV REQUIRED NO_DEFAULT_PATH HINTS ${opencv_3_1_path} )
#find_package( OpenCV REQUIRED)
add_executable( TestSolvePnp TestSolvePnp.cpp epnp/epnp.cpp)
target_link_libraries( TestSolvePnp ${OpenCV_LIBS} )
