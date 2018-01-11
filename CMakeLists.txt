cmake_minimum_required(VERSION 2.8)
project(rl-pilot)
#find_package( CUDA REQUIRED )
find_package( OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11  -Wall -fPIC")
set (PPRZLINK_INCL_DIR ./pprzlink/var/include/)
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3 -std=c++11 -pthread")

#include_directories(./../tensorflow)
#include_directories(./../tensorflow/bazel-tensorflow/external/eigen_archive/)
#include_directories(./../tensorflow/bazel-tensorflow/external/eigen_archive/eigen-eigen-3f653ace7d28/)
#include_directories(./../tensorflow/bazel-genfiles)
#include_directories(/usr/local/lib/python3.5/dist-packages/tensorflow/include/external/nsync/public/)
#include_directories(/usr/local/cuda/lib/)
include_directories("${PPRZLINK_INCL_DIR}")
include_directories(./udp)

FILE(GLOB PPRZLINK_SRC_FILES "./pprzlink/var/share/pprzlink/src/pprz_transport.c")
FILE(GLOB UDP_SRC_FILES "./udp/*.*")
add_executable(rlp ${UDP_SRC_FILES}  rlpilot.cpp ${PPRZLINK_SRC_FILES} datalink.cpp)  #


# link_directories(${CUDA_LIBRARIES})
# link_directories(/usr/local/cuda-6.5/lib/)
# target_link_libraries(rlp -rpath-link/usr/local/cuda-6.5/lib/)
#target_link_libraries(rlp -L/usr/local/cuda-6.5/lib/ -lcudnn)
#target_link_libraries(rlp -L/usr/local/cuda-6.5/lib/ -lcudart)
#target_link_libraries(rlp -L./../tensorflow/bazel-bin/tensorflow -ltensorflow_cc)
#target_link_libraries(rlp -L/usr/lib/ -lopencv_core)
#target_link_libraries(rlp -L/usr/lib/ -lopencv_imgproc)
#target_link_libraries(rlp -L/usr/lib/ -lopencv_highgui)
#target_link_libraries(rlp -L/usr/lib -lkalamos-context)

target_link_libraries( rlp ${OpenCV_LIBS} )