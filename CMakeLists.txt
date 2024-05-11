cmake_minimum_required(VERSION 3.10)

# set the project name
project(DRTMCS VERSION 0.1 LANGUAGES CXX CUDA)

# Cpp standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall -O0 -g -lstdc++fs")

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

# Add the Google Test submodule directory
add_subdirectory(external/googletest)

# Include Google Test in the project
enable_testing()
include(GoogleTest)

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
        src/cuda/Volume.cu
        src/cuda/kernel.cu
        src/cuda/CudaSurface3D.cu
        src/cuda/VolumeRendering.cu
        src/cuda/PlaneCutRendering.cu src/utils/filesystem.h src/model/AdamOptimizer.cpp src/model/AdamOptimizer.hpp src/view/ImUI/Inspectors/AdamEditor.hpp src/interactors/AdamInteractor.cpp src/interactors/AdamInteractor.hpp src/cuda/Losses.cuh src/cuda/Convolutions.cuh src/model/Gaussian/GPUGaussianKernelData.cpp src/model/Gaussian/GPUGaussianKernelData.hpp src/cuda/Convolutions.cu src/model/DataLoader/DataLoader.cpp src/model/DataLoader/DataLoader.hpp src/cuda/IntegrationRange.cu src/cuda/IntegrationRange.cuh src/cuda/Adam.cu src/cuda/Adam.cuh src/model/Distribution/NormalDistributionClamped.hpp src/model/SuperResolution/SuperResolutionModule.cpp src/model/SuperResolution/SuperResolutionModule.h src/model/Distribution/UniformDistribution.hpp src/cuda/SparseVolumeUtils.cu src/cuda/SparseVolumeUtils.cuh src/model/Volume/SparseVolume3D.cpp src/model/Volume/SparseVolume3D.hpp src/cuda/BitUtils.cuh
        src/model/Statistics/Statistics.cpp
        src/model/Statistics/Statistics.h
        src/model/Statistics/StatsType.h
        src/model/Volume/Volume3D.h
        src/view/PointCloud.cpp
        src/view/PointCloud.h
        src/view/ImUI/StatsView.cpp
        src/view/ImUI/StatsView.h
        src/model/Statistics/ScrollingBuffer.h
        src/model/Octree/OctreeStructs.h
        src/model/Octree/SVO.cpp
        src/model/Octree/SVO.h
)

set(CUDA_ARCH "-arch=sm_89")
set(CMAKE_CUDA_ARCHITECTURES "89")

set(SOURCE_FILES
        src/main.cpp

        include/imgui/imgui.cpp
        include/imgui/imgui_widgets.cpp
        include/imgui/imgui_draw.cpp
        include/imgui/imgui_demo.cpp
        include/imgui/imgui_tables.cpp
        include/imgui/backends/imgui_impl_glfw.cpp
        include/imgui/backends/imgui_impl_opengl3.cpp

        include/implot/implot.cpp
        include/implot/implot_items.cpp
        include/implot/implot_demo.cpp

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
        src/model/Volume/DenseVolume3D.cpp
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
        src/view/PointCloud.cpp

        src/utils/SceneSettings.cpp
)
# add the executable
cuda_add_executable(DRTMCS ${SOURCE_FILES} ${CUDA_SOURCE_FILES})

target_compile_options(DRTMCS PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        -g
        -G
        --relocatable-device-code=true
        >)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("include/")
include_directories("include/tomlpp")
include_directories("include/imgui")
include_directories("include/implot")
include_directories("include/imgui/backends")
include_directories("include/glfw/include")
include_directories("src/cuda/")
include_directories("src/utils/")
include_directories("external/libmorton/include")

message("Cuda include dirs: ")
message(${CUDA_INCLUDE_DIRS})

include_directories(${CUDA_INCLUDE_DIRS})

# OpenEXR::OpenEXR
# target_link_libraries(ProjectName PRIVATE libraw::libraw)
#libraw::libraw
target_link_libraries(DRTMCS nlohmann_json::nlohmann_json OpenGL::GL GLEW::GLEW glfw glm assimp ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${OPENGL_LIBRARIES})


# Test project
set(TEST_FILES
        tests/main.test.cpp
)
add_executable(DRTMCSTests ${TEST_FILES})
target_link_libraries(DRTMCSTests gtest gtest_main glm assimp ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${OPENGL_LIBRARIES})
gtest_discover_tests(DRTMCSTests)

