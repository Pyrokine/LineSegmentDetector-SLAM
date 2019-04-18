cmake_minimum_required(VERSION 2.8.3)
project(lsd)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  geometry_msgs
)
find_package(OpenCV)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    sensor_msgs
    geometry_msgs
  INCLUDE_DIRS include
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(lsd 
                  src/main_on_linux.cpp
                  src/baseFunc.cpp
                  src/FeatureAssociation.cpp
                  src/myLSD.cpp
                  src/myRDP.cpp)
target_link_libraries(lsd ${catkin_LIBRARIES})
target_link_libraries(lsd ${OpenCV_LIBS})
