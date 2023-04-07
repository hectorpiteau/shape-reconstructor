cmake_minimum_required(VERSION 3.10)

# set the project name
project(DRTMCS VERSION 0.1 LANGUAGES CXX CUDA)

# Cpp standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# set(CMAKE_CXX_FLAGS "-std=c++14 -lstdc++fs" )

# GLFW Settings 
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# JSON
# Typically you don't care so much for a third party library's tests to be run from your own project's code.
set(JSON_BuildTests OFF CACHE INTERNAL "")

# Add subdirectories (gits)
add_subdirectory(include/nlohmann)
add_subdirectory(include/glfw)
add_subdirectory(include/assimp)
add_subdirectory(include/glm)

# find_package(glfw3 3.3 REQUIRED)
# find_package(assimp REQUIRED)
# find_package(OpenEXR REQUIRED)

find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)

#Glew
find_package(GLEW REQUIRED)
# find_package(glm REQUIRED)

#CUDA
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
find_package(CUDA REQUIRED)
enable_language(CUDA)
set(CUDA_SOURCE_FILES src/cuda/kernel.cu src/cuda/CudaSurface3D.cu src/cuda/VolumeRendering.cu)
set(CUDA_ARCH "-arch=sm_61")
set(CMAKE_CUDA_ARCHITECTURES "61")


# add the executable
cuda_add_executable(DRTMCS 
    src/main.cpp 
    
    include/imgui/imgui.cpp
    include/imgui/imgui_widgets.cpp
    include/imgui/imgui_draw.cpp
    include/imgui/imgui_demo.cpp
    include/imgui/imgui_tables.cpp
    include/imgui/backends/imgui_impl_glfw.cpp
    include/imgui/backends/imgui_impl_opengl3.cpp
    
    src/view/ImUI/ObjectListView.cpp
    src/view/ImUI/ObjectListItem.cpp

    src/interactors/ObjectListInteractor.cpp 

    src/controllers/Scene/Scene.cpp 

    src/model/ShaderPipeline.cpp 
    src/model/Camera.cpp 
    # src/model/SphereSDF.cpp
    src/model/DenseFloat32Volume.cpp

    src/model/Dataset/NeRFDataset.cpp

    src/view/UnitCube.cpp 
    src/view/Grid.cpp 
    src/view/SkyBox.cpp 
    src/view/OverlayPlane.cpp 
    src/view/Gizmo.cpp 
    src/view/Lines.cpp 
    src/view/Volume3D.cpp 
    src/view/LineGrid.cpp 
    src/view/ImageSet.cpp 

    ${CUDA_SOURCE_FILES})

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories("include/")
include_directories("include/imgui")
include_directories("include/imgui/backends")
include_directories("include/glfw/include")
include_directories("src/cuda/")


message("Cuda include dirs: ")
message(${CUDA_INCLUDE_DIRS})

include_directories(${CUDA_INCLUDE_DIRS})

# OpenEXR::OpenEXR
target_link_libraries(DRTMCS nlohmann_json::nlohmann_json OpenGL::GL GLEW::GLEW glfw glm assimp ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${OPENGL_LIBRARIES})
