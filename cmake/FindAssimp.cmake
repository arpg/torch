# - Try to find Assimp
# Once done, this will define
#
#  Assimp_FOUND - system has Assimp
#  Assimp_INCLUDE_DIR - the Assimp include directories
#  Assimp_LIBRARIES - link these to use Assimp

find_path(Assimp_INCLUDE_DIR NAMES assimp/mesh.h)
find_library(Assimp_LIBRARIES NAMES assimp)
include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
  Assimp
  DEFAULT_MSG
  Assimp_INCLUDE_DIR
  Assimp_LIBRARIES
)

set(Glog_INCLUDE_DIRS ${Assimp_INCLUDE_DIRS})
set(Glog_LIBRARIES ${Assimp_LIBRARIES})
