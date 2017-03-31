cmake_minimum_required(VERSION 2.8)
project(egbis_builder C CXX)
include(ExternalProject)

ExternalProject_Add(egbis
    GIT_REPOSITORY https://github.com/mingkaic/opencv-wrapper-egbis.git
    CMAKE_ARGS -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
    -DCMAKE_CXX_FLAGS=${MSVC_COMPILER_DEFS}
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}")

# Specify include dir and link libraries
ExternalProject_Get_Property(egbis source_dir)
set(EGBIS_INCLUDE_DIRS ${source_dir}/egbis/include PARENT_SCOPE)
set(EGBIS_LIBS_DIR ${source_dir}/bin/libegbis.a PARENT_SCOPE)
