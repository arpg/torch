add_executable(main main.cpp)
target_link_libraries(main ${LIBRARY_NAME})
add_dependencies(main ${LIBRARY_NAME})