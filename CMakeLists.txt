#>>>>>>>>>>>>>>>>>>>>>>>> GENERAL SETUP COMMENCE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
cmake_minimum_required(VERSION 2.8)
set(PROJECT_NAME_STR logoRNN)
project(${PROJECT_NAME_STR} C CXX)

set(EXT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ext)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(${EXT_PATH}/egbis)

set(CMAKE_BINARY_DIR ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#-------------------
# set compilation flag
#-------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED on)

set(EXT_PROJECTS_DIR ${PROJECT_SOURCE_DIR}/ext)
set(LOGO_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
FILE(GLOB_RECURSE LOGO_SRC src/*.cpp)
FILE(GLOB_RECURSE LOGO_TOOLS tools/*.cpp)

include_directories(${LOGO_INCLUDE} ${EGBIS_INCLUDE_DIRS})
add_executable(${PROJECT_NAME_STR} ${LOGO_SRC} ${LOGO_TOOLS})
add_dependencies(${PROJECT_NAME_STR} egbis)
target_link_libraries(${PROJECT_NAME_STR} ${OpenCV_LIBS} ${EGBIS_LIBS_DIR})
