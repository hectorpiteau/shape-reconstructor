cmake_minimum_required(VERSION 3.10)

# set the project name
project(DRTMCS VERSION 0.1 LANGUAGES CXX CUDA)

# Cpp standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall -O0 -lstdc++fs")

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
#add_subdirectory(include/libraw)

# find_package(glfw3 3.3 REQUIRED)
# find_package(assimp REQUIRED)
# find_package(OpenEXR REQUIRED)

find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)

#Glew
find_package(GLEW REQUIRED)
# find_package(glm REQUIRED)

#CUDA
if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()

find_package(CUDA REQUIRED)

enable_language(CUDA)

set(CUDA_SOURCE_FILES
        src/cuda/kernel.cu
        src/cuda/CudaSurface3D.cu
        src/cuda/VolumeRendering.cu
        src/cuda/PlaneCutRendering.cu src/utils/filesystem.h src/model/AdamOptimizer.cpp src/model/AdamOptimizer.hpp src/view/ImUI/Inspectors/AdamEditor.hpp src/interactors/AdamInteractor.cpp src/interactors/AdamInteractor.hpp src/cuda/Losses.cuh src/cuda/Convolutions.cuh src/model/Gaussian/GPUGaussianKernelData.cpp src/model/Gaussian/GPUGaussianKernelData.hpp src/cuda/Convolutions.cu src/model/DataLoader/DataLoader.cpp src/model/DataLoader/DataLoader.hpp src/cuda/IntegrationRange.cu src/cuda/IntegrationRange.cuh)
set(CUDA_ARCH "-arch=sm_75")
set(CMAKE_CUDA_ARCHITECTURES "75")


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

        src/view/ImUI/ObjectListItem.cpp
        src/view/ImUI/ObjectListView.cpp
        src/view/ImUI/InspectorView.cpp

        src/interactors/ImageSetInteractor.cpp
        src/interactors/CameraInteractor.cpp
        src/interactors/ObjectListInteractor.cpp
        src/interactors/SceneObjectInteractor.cpp
        src/interactors/NeRFInteractor.cpp
        src/interactors/CameraSetInteractor.cpp
        src/interactors/Volume3DInteractor.cpp
        src/interactors/VolumeRendererInteractor.cpp
        src/interactors/SimpleRayCasterInteractor.cpp
        src/interactors/PlaneCutInteractor.cpp

        src/controllers/Scene/Scene.cpp

        src/model/ShaderPipeline.cpp
        src/model/Camera/Camera.cpp
        src/model/Camera/CameraSet.cpp
        # src/model/SphereSDF.cp
        src/model/DenseFloat32Volume.cpp
        src/model/Dataset/NeRFDataset.cpp
        src/model/ImageSet.cpp
        src/model/Volume3D.cpp
        src/model/Mesh.cpp
        src/model/VolumeRenderer.cpp
        src/model/PlaneCut.cpp

        # src/model/RayCaster/SingleRayCaster.cpp

        src/view/UnitCube.cpp
        src/view/Grid.cpp
        src/view/SkyBox.cpp
        src/view/OverlayPlane.cpp
        src/view/Plane.cpp
        src/view/Gizmo.cpp
        src/view/Lines.cpp
        src/view/LineGrid.cpp

        ${CUDA_SOURCE_FILES})

target_compile_options(DRTMCS PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -G
        # --use_fast_math
        --relocatable-device-code=true
        >)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("include/")
include_directories("include/tomlpp")
include_directories("include/imgui")
include_directories("include/imgui/backends")
include_directories("include/glfw/include")
include_directories("src/cuda/")
include_directories("src/utils/")


message("Cuda include dirs: ")
message(${CUDA_INCLUDE_DIRS})

include_directories(${CUDA_INCLUDE_DIRS})

# OpenEXR::OpenEXR
# target_link_libraries(ProjectName PRIVATE libraw::libraw)
#libraw::libraw
target_link_libraries(DRTMCS nlohmann_json::nlohmann_json OpenGL::GL GLEW::GLEW glfw glm assimp ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${OPENGL_LIBRARIES})


