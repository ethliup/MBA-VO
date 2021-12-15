file(GLOB FILES_BLUR_TRACKER_SRCS
    "*.cpp"
)

add_library(slam_blur_tracker ${FILES_BLUR_TRACKER_SRCS})
target_link_libraries(slam_blur_tracker slam_core slam_utils)

if (CUDA_FOUND)
    file(GLOB FILES_VO_BLUR_TRACKER_CUDA
        "*.cu")
        
    cuda_add_library(cuda_blur_tracker 
        ${FILES_VO_BLUR_TRACKER_CUDA}
        OPTIONS 
        -std c++11)

    target_link_libraries(cuda_blur_tracker ${CUDA_LIBRARIES} slam_core)
    target_link_libraries(slam_blur_tracker cuda_blur_tracker)
endif()