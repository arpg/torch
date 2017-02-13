add_executable(solver solver.cpp)
target_link_libraries(solver ${LIBRARY_NAME})
add_dependencies(solver ${LIBRARY_NAME})