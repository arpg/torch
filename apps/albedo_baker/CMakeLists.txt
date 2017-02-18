add_executable(albedo_baker albedo_baker.cpp)
target_link_libraries(albedo_baker ${LIBRARY_NAME})
add_dependencies(albedo_baker ${LIBRARY_NAME})
