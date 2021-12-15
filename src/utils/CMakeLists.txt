file(GLOB FILES_UTILS_HEADERS
        "*.h"
        "*.hpp"
)

file(GLOB FILES_UTILS_SRCS
        "*.cpp"
)

set(FILES_UTILS_ALL
    ${FILES_UTILS_HEADERS}
    ${FILES_UTILS_SRCS})

add_library(slam_utils ${FILES_UTILS_ALL})
target_link_libraries(slam_utils slam_core)