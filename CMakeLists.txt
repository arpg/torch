cmake_minimum_required(VERSION 2.8)

# set project properties

project(Torch)
string(TOLOWER ${PROJECT_NAME} LIBRARY_NAME)

# set compilers arguments

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# find required packages

find_package(CUDA REQUIRED)
list(APPEND REQUIRED_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
list(APPEND REQUIRED_LIBRARIES ${CUDA_LIBRARIES})

find_package(OptiX REQUIRED)
list(APPEND REQUIRED_INCLUDE_DIRS ${OptiX_INCLUDE_DIRS})
list(APPEND REQUIRED_LIBRARIES ${OptiX_LIBRARIES})

find_package(OpenCV REQUIRED)
list(APPEND REQUIRED_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
list(APPEND REQUIRED_LIBRARIES ${OpenCV_LIBRARIES})

find_package(GFlags REQUIRED)
list(APPEND REQUIRED_INCLUDE_DIRS ${GFlags_INCLUDE_DIRS})
list(APPEND REQUIRED_LIBRARIES ${GFlags_LIBRARIES})

find_package(GLog REQUIRED)
list(APPEND REQUIRED_INCLUDE_DIRS ${GLog_INCLUDE_DIRS})
list(APPEND REQUIRED_LIBRARIES ${GLog_LIBRARIES})

list(APPEND REQUIRED_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include)
list(APPEND REQUIRED_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/include)
include_directories(${REQUIRED_INCLUDE_DIRS})

# define config headers

set(CONFIG_HEADER_DIR include/${LIBRARY_NAME})

set(CONFIG_HEADERS
  ${CONFIG_HEADER_DIR}/${PROJECT_NAME}.h
  ${CONFIG_HEADER_DIR}/PtxUtil.h
)

# define c++ headers

set(CPP_HEADER_DIR include/${LIBRARY_NAME})

set(CPP_HEADERS
  ${CPP_HEADER_DIR}/Camera.h
  ${CPP_HEADER_DIR}/Context.h
  ${CPP_HEADER_DIR}/Object.h
  ${CPP_HEADER_DIR}/Node.h
  ${CPP_HEADER_DIR}/Ray.h
  ${CPP_HEADER_DIR}/Scene.h
)

# define c++ sources

set(CPP_SOURCE_DIR src)

set(CPP_SOURCES
  ${CPP_SOURCE_DIR}/Camera.cpp
  ${CPP_SOURCE_DIR}/Context.cpp
  ${CPP_SOURCE_DIR}/Object.cpp
  ${CPP_SOURCE_DIR}/Node.cpp
  ${CPP_SOURCE_DIR}/Scene.cpp
)

# define cuda headers

set(CUDA_HEADER_DIR include/${LIBRARY_NAME})

set(CUDA_HEADERS
)

# define cuda sources

set(CUDA_SOURCE_DIR src)

set(CUDA_SOURCES
  ${CUDA_SOURCE_DIR}/Camera.cu
)

# set project definitions

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_definitions("-DDEBUG_BUILD")
endif()

# build project main header

foreach(CPP_HEADER ${CPP_HEADERS})
  string(REPLACE "include/" "" CPP_HEADER ${CPP_HEADER})
  set(PROJECT_INCLUDES "${PROJECT_INCLUDES}\n#include <${CPP_HEADER}>")
endforeach()

# build config headers

set(PTX_FOLDER "${CMAKE_BINARY_DIR}/")
set(PTX_PREFIX "cuda_compile_ptx_generated_")
set(PTX_SUFFIX ".cu.ptx")

foreach(CONFIG_HEADER ${CONFIG_HEADERS})
  configure_file(${CONFIG_HEADER}.in ${CONFIG_HEADER})
endforeach()

# build ptx files

cuda_compile_ptx(PTX_FILES ${CUDA_SOURCES})

add_custom_target(PTX_FILES ALL
  DEPENDS ${PTX_FILES} ${CUDA_SOURCES}
  SOURCES ${CUDA_SOURCES}
)

# build project library

add_library(${LIBRARY_NAME} SHARED ${CPP_SOURCES} ${CPP_HEADERS})
target_link_libraries(${LIBRARY_NAME} ${REQUIRED_LIBRARIES})

# build applications

option(BUILD_APPS "Build applications" ON)

if(BUILD_APPS)
  add_subdirectory(apps)
endif()

# build unit tests

option(BUILD_TESTS "Build unit tests" OFF)

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()