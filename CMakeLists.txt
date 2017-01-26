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
  ${CPP_HEADER_DIR}/Distribution.h
  ${CPP_HEADER_DIR}/Exception.h
  ${CPP_HEADER_DIR}/Geometry.h
  ${CPP_HEADER_DIR}/GeometryGroup.h
  ${CPP_HEADER_DIR}/Group.h
  ${CPP_HEADER_DIR}/Image.h
  ${CPP_HEADER_DIR}/Light.h
  ${CPP_HEADER_DIR}/LightData.h
  ${CPP_HEADER_DIR}/LightSampler.h
  ${CPP_HEADER_DIR}/Link.h
  ${CPP_HEADER_DIR}/Object.h
  ${CPP_HEADER_DIR}/PointLight.h
  ${CPP_HEADER_DIR}/PointLightSampler.h
  ${CPP_HEADER_DIR}/Primitive.h
  ${CPP_HEADER_DIR}/Material.h
  ${CPP_HEADER_DIR}/MatteMaterial.h
  ${CPP_HEADER_DIR}/Node.h
  ${CPP_HEADER_DIR}/Ray.h
  ${CPP_HEADER_DIR}/Scene.h
  ${CPP_HEADER_DIR}/SceneLightSampler.h
  ${CPP_HEADER_DIR}/SingleGeometry.h
  ${CPP_HEADER_DIR}/Spectrum.h
  ${CPP_HEADER_DIR}/Sphere.h
  ${CPP_HEADER_DIR}/Transform.h
  ${CPP_HEADER_DIR}/Transformable.h
)

# define c++ sources

set(CPP_SOURCE_DIR src)

set(CPP_SOURCES
  ${CPP_SOURCE_DIR}/Camera.cpp
  ${CPP_SOURCE_DIR}/Context.cpp
  ${CPP_SOURCE_DIR}/Distribution.cpp
  ${CPP_SOURCE_DIR}/Geometry.cpp
  ${CPP_SOURCE_DIR}/GeometryGroup.cpp
  ${CPP_SOURCE_DIR}/Group.cpp
  ${CPP_SOURCE_DIR}/Image.cpp
  ${CPP_SOURCE_DIR}/Light.cpp
  ${CPP_SOURCE_DIR}/LightSampler.cpp
  ${CPP_SOURCE_DIR}/Link.cpp
  ${CPP_SOURCE_DIR}/Object.cpp
  ${CPP_SOURCE_DIR}/PointLight.cpp
  ${CPP_SOURCE_DIR}/PointLightSampler.cpp
  ${CPP_SOURCE_DIR}/Primitive.cpp
  ${CPP_SOURCE_DIR}/Material.cpp
  ${CPP_SOURCE_DIR}/MatteMaterial.cpp
  ${CPP_SOURCE_DIR}/Node.cpp
  ${CPP_SOURCE_DIR}/Scene.cpp
  ${CPP_SOURCE_DIR}/SceneLightSampler.cpp
  ${CPP_SOURCE_DIR}/SingleGeometry.cpp
  ${CPP_SOURCE_DIR}/Spectrum.cpp
  ${CPP_SOURCE_DIR}/Sphere.cpp
  ${CPP_SOURCE_DIR}/Transform.cpp
  ${CPP_SOURCE_DIR}/Transformable.cpp
)

# define cuda headers

set(CUDA_HEADER_DIR include/${LIBRARY_NAME})

set(CUDA_HEADERS
  ${CUDA_HEADER_DIR}/Geometry.cuh
)

# define cuda sources

set(CUDA_SOURCE_DIR src)

set(CUDA_SOURCES
  ${CUDA_SOURCE_DIR}/Camera.cu
  ${CUDA_SOURCE_DIR}/Distribution.cu
  ${CUDA_SOURCE_DIR}/MatteMaterial.cu
  ${CUDA_SOURCE_DIR}/PointLightSampler.cu
  ${CUDA_SOURCE_DIR}/SceneLightSampler.cu
  ${CUDA_SOURCE_DIR}/Sphere.cu
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