add_executable(solver solver.cpp)
target_link_libraries(solver ${LIBRARY_NAME})
add_dependencies(solver ${LIBRARY_NAME})

add_executable(albedo_solver albedo_solver.cpp)
target_link_libraries(albedo_solver ${LIBRARY_NAME})
add_dependencies(albedo_solver ${LIBRARY_NAME})
