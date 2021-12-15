set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/)

add_executable(test_blur_aware_tracker_modules test_blur_aware_tracker_modules.cpp)
target_link_libraries(test_blur_aware_tracker_modules slam_core slam_blur_tracker)
if(CUDA_FOUND)
    target_link_libraries(test_blur_aware_tracker_modules cuda_blur_tracker)
endif()

