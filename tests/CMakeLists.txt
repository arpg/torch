enable_testing()

# find required packages

find_package(GTest REQUIRED)
list(APPEND REQUIRED_TEST_INCLUDE_DIRS ${GTEST_INCLUDE_DIRS})
list(APPEND REQUIRED_TEST_LIBRARIES ${GTEST_LIBRARIES})

find_package(Eigen3 REQUIRED)
list(APPEND REQUIRED_TEST_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

list(APPEND REQUIRED_TEST_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/tests)
list(APPEND REQUIRED_TEST_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/tests)
include_directories(${REQUIRED_TEST_INCLUDE_DIRS})

# define config headers

set(TEST_CONFIG_HEADER_DIR .)

set(TEST_CONFIG_HEADERS
  ${TEST_CONFIG_HEADER_DIR}/TestPtxUtil.h
)

# define c++ headers

set(TEST_CPP_HEADER_DIR .)

set(TEST_CPP_HEADERS
)

# define c++ sources

set(TEST_CPP_SOURCE_DIR .)

set(TEST_CPP_SOURCES
  ${TEST_CPP_SOURCE_DIR}/ActivationCostFunction.cpp
  ${TEST_CPP_SOURCE_DIR}/AlbedoCostFunction.cpp
  ${TEST_CPP_SOURCE_DIR}/Distribution1D.cpp
  ${TEST_CPP_SOURCE_DIR}/Distribution2D.cpp
  ${TEST_CPP_SOURCE_DIR}/LightCostFunction.cpp
  ${TEST_CPP_SOURCE_DIR}/MeshWriter.cpp
  ${TEST_CPP_SOURCE_DIR}/Node.cpp
  ${TEST_CPP_SOURCE_DIR}/Problem.cpp
  ${TEST_CPP_SOURCE_DIR}/ReflectanceCostFunction.cpp
  ${TEST_CPP_SOURCE_DIR}/Sampling.cpp
  ${TEST_CPP_SOURCE_DIR}/Transform.cpp
)

# define cuda sources

set(TEST_CUDA_SOURCE_DIR .)

set(TEST_CUDA_SOURCES
  ${TEST_CUDA_SOURCE_DIR}/Distribution1D.cu
  ${TEST_CUDA_SOURCE_DIR}/Distribution2D.cu
)

# build config headers

set(TEST_PTX_FOLDER "${CMAKE_BINARY_DIR}/tests/")
set(TEST_PTX_PREFIX "cuda_compile_ptx_generated_")
set(TEST_PTX_SUFFIX ".cu.ptx")

foreach(TEST_CONFIG_HEADER ${TEST_CONFIG_HEADERS})
  configure_file(${TEST_CONFIG_HEADER}.in ${TEST_CONFIG_HEADER})
endforeach()

# build ptx files

cuda_compile_ptx(PTX_FILES ${TEST_CUDA_SOURCES})

add_custom_target(TEST_PTX_FILES ALL
  DEPENDS ${TEST_PTX_FILES} ${TEST_CUDA_SOURCES}
  SOURCES ${TEST_CUDA_SOURCES}
)

# build test programs

add_executable(tests tests.cpp ${TEST_CPP_SOURCES} ${TEST_CPP_HEADERS})
target_link_libraries(tests ${LIBRARY_NAME} ${REQUIRED_TEST_LIBRARIES})
add_dependencies(tests ${LIBRARY_NAME})
