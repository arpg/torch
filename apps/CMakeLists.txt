add_executable(main main.cpp)
target_link_libraries(main ${LIBRARY_NAME})
add_dependencies(main ${LIBRARY_NAME})

add_executable(optim optim.cpp)
target_link_libraries(optim ${LIBRARY_NAME})
add_dependencies(optim ${LIBRARY_NAME})

configure_file(scene.xml scene.xml COPYONLY)
configure_file(bunny.ply bunny.ply COPYONLY)
configure_file(shark.ply shark.ply COPYONLY)
