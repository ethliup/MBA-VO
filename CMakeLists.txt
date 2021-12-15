cmake_minimum_required(VERSION 3.5)
project(BA_tracker)

# ========================================================================== #
#                      COMPILATION FLAGS CONFIG                              #
# ========================================================================== #
# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  #add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "^MSVC$")
  # under windows, explicitly enable math defines like M_PI
  # allow larget objects than default
  # enable parallel compilation
  add_definitions("-D _USE_MATH_DEFINES /MP /fp:fast /INCREMENTAL")
  add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
endif()

# =========================================================================== #
#                                 Find Packages                               #
# =========================================================================== #
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# ========================================================================== #
#                        Add package include/lib path                        # 
# ========================================================================== #
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/third_party/cub)

if(CUDA_FOUND)
  MESSAGE(STATUS "Found CUDA verision: " ${CUDA_VERSION_STRING} " installed in: " ${CUDA_TOOLKIT_ROOT_DIR})
  include_directories(${CUDA_INCLUDE_DIRS})
  add_definitions(-DCOMPILE_WITH_CUDA)
endif()

# ========================================================================== #
#                             Build libs and targets                         # 
# ========================================================================== #
add_subdirectory(src)
add_subdirectory(test)

