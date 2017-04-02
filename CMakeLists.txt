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

#========= OBJECT PROPOSAL LIBRARY =========#
set(PROP_PROJ objprop)
set(LOGO_INCLUDE proposal/include)
FILE(GLOB_RECURSE LOGO_SRC proposal/src/*.cpp)
include_directories(${LOGO_INCLUDE} ${EGBIS_INCLUDE_DIRS})
add_library(${PROP_PROJ} ${LOGO_SRC})
add_dependencies(${PROP_PROJ} egbis)
target_link_libraries(${PROP_PROJ} ${OpenCV_LIBS} ${EGBIS_LIBS_DIR})

#========= OBJECT PROPOSAL TEST EXECUTABLE =========#
set(PROP_TEST optest)
include_directories(${LOGO_INCLUDE})
add_executable(${PROP_TEST} tools/test_objprop.cpp)
target_link_libraries(${PROP_TEST} ${OpenCV_LIBS} ${PROP_PROJ})

#========= FAST R-CNN EXECUTABLE =========#
set(PROP_EXEC opgen)
include_directories(${LOGO_INCLUDE})
add_executable(${PROP_EXEC} tools/opgen.cpp)
target_link_libraries(${PROP_EXEC} ${OpenCV_LIBS} ${PROP_PROJ})