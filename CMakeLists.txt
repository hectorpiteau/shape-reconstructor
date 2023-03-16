cmake_minimum_required(VERSION 3.10)

# set the project name
project(DRTMCVFX VERSION 0.1 LANGUAGES CXX CUDA)

# Cpp standards
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# GLFW Settings 
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

find_package(glfw3 3.3 REQUIRED)
find_package(assimp REQUIRED)
find_package(OpenEXR REQUIRED)

find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)

#Glew
find_package(GLEW REQUIRED)
find_package(glm REQUIRED)

#CUDA
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
find_package(CUDA REQUIRED)
enable_language(CUDA)
set(CUDA_SOURCE_FILES src/cuda/kernel.cu)
set(CUDA_ARCH "-arch=sm_61")
set(CMAKE_CUDA_ARCHITECTURES "61")


# add the executable
cuda_add_executable(DRTMCVFX 
    src/main.cpp 
    src/model/ShaderPipeline.cpp 
    src/model/Camera.cpp 
    src/view/UnitCube.cpp 
    src/view/Grid.cpp 
    src/view/SkyBox.cpp 
    src/view/OverlayPlane.cpp 
    src/cuda/assert_cuda.c 
    # interop.c 
    ${CUDA_SOURCE_FILES})

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories("include/")
include_directories("src/cuda/")


message("Cuda include dirs: ")
message(${CUDA_INCLUDE_DIRS})

include_directories(${CUDA_INCLUDE_DIRS})

target_link_libraries(DRTMCVFX OpenGL::GL GLEW::GLEW glfw assimp OpenEXR::OpenEXR ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${OPENGL_LIBRARIES})
