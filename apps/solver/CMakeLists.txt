add_executable(solver solver.cpp)
target_link_libraries(solver ${LIBRARY_NAME})
add_dependencies(solver ${LIBRARY_NAME})

add_executable(albedo_solver albedo_solver.cpp)
target_link_libraries(albedo_solver ${LIBRARY_NAME})
add_dependencies(albedo_solver ${LIBRARY_NAME})

add_executable(voxel_solver voxel_solver.cpp)
target_link_libraries(voxel_solver ${LIBRARY_NAME})
add_dependencies(voxel_solver ${LIBRARY_NAME})

add_executable(mesh_solver mesh_solver.cpp)
target_link_libraries(mesh_solver ${LIBRARY_NAME})
add_dependencies(mesh_solver ${LIBRARY_NAME})