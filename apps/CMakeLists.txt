add_executable(main main.cpp)
target_link_libraries(main ${LIBRARY_NAME})
add_dependencies(main ${LIBRARY_NAME})

add_executable(image_processor image_processor.cpp)
target_link_libraries(image_processor ${LIBRARY_NAME})
add_dependencies(image_processor ${LIBRARY_NAME})

add_executable(voxel_demo voxel_demo.cpp)
target_link_libraries(voxel_demo ${LIBRARY_NAME})
add_dependencies(voxel_demo ${LIBRARY_NAME})

add_executable(mesh_solver_test mesh_solver_test.cpp)
target_link_libraries(mesh_solver_test ${LIBRARY_NAME})
add_dependencies(mesh_solver_test ${LIBRARY_NAME})

find_package(Eigen3 QUIET)

if(EIGEN3_FOUND)
  include_directories(${EIGEN3_INCLUDE_DIR})
  add_executable(pose_processor pose_processor.cpp)
  target_link_libraries(pose_processor ${LIBRARY_NAME})
  add_dependencies(pose_processor ${LIBRARY_NAME})
endif()

add_subdirectory(albedo_baker)

if(Lynx_FOUND)
  add_subdirectory(solver)
endif()

configure_file(reference.png reference.png COPYONLY)
configure_file(scene.xml scene.xml COPYONLY)
configure_file(bunny.ply bunny.ply COPYONLY)
configure_file(shark.ply shark.ply COPYONLY)