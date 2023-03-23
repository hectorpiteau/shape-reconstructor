#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <tgmath.h>

#include <sstream>
#include "model/Camera.hpp"
#include "model/ShaderPipeline.hpp"
#include "model/Model.hpp"
#include "model/CudaTexture.hpp"
#include "maths/MMath.hpp"
#include "utils/FileUtils.hpp"
#include "utils/Utils.hpp"
#include "view/UnitCube.hpp"
#include "view/Grid.hpp"
#include "view/SkyBox.hpp"
#include "view/OverlayPlane.hpp"
#include "view/Lines.hpp"
#include "view/Gizmo.hpp"
#include "view/Volume3D.hpp"
#include "view/LineGrid.hpp"
#include "utils/SceneSettings.hpp"
#include "utils/Projection.hpp"

#include "../include/imgui/imgui.h"
#include "../include/imgui/backends/imgui_impl_glfw.h"
#include "../include/imgui/backends/imgui_impl_opengl3.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../include/stb_image.h"

using namespace cv;

using namespace glm;

#define GLSL(src) #src
#define WINDOW_WIDTH 1080
#define WINDOW_HEIGHT 720

float FOV = 90.0f;
float NEAR = 1.0f;
float FAR = 10.0f;

std::shared_ptr<SceneSettings> sceneSettings = std::make_shared<SceneSettings>(1080, 720);

static void pxl_glfw_fps(GLFWwindow *window)
{
    // static fps counters
    static double stamp_prev = 0.0;
    static int frame_count = 0;

    // locals
    const double stamp_curr = glfwGetTime();
    const double elapsed = stamp_curr - stamp_prev;

    if (elapsed > 0.5)
    {
        stamp_prev = stamp_curr;

        const double fps = (double)frame_count / elapsed;

        int width, height;
        char tmp[64];

        std::ostringstream oss;
        glfwGetFramebufferSize(window, &width, &height);

        oss << "(" << width << "," << height << ") - FPS: " << fps;

        glfwSetWindowTitle(window, oss.str().c_str());

        frame_count = 0;
    }

    frame_count++;
}

static void render(GLFWwindow *window, GLuint shaderProgram)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_ACCUM_BUFFER_BIT);
}

static void error_callback(int error, const char *description)
{
    fputs(description, stderr);
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    // don't pass mouse and keyboard presses further if an ImGui widget is active
    auto& io = ImGui::GetIO();
    if (io.WantCaptureMouse || io.WantCaptureKeyboard) {
        return;
    }

    sceneSettings->Scroll(xoffset, yoffset);
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    // don't pass mouse and keyboard presses further if an ImGui widget is active
    auto& io = ImGui::GetIO();
    if (io.WantCaptureMouse || io.WantCaptureKeyboard) {
        return;
    }

    if (action == GLFW_PRESS)
    {
        switch (button)
        {
        case GLFW_MOUSE_BUTTON_LEFT:
            sceneSettings->SetMouseLeftClick(true);
            break;
        case GLFW_MOUSE_BUTTON_RIGHT:
            sceneSettings->SetMouseRightClick(true);
            break;
        }
    }
    else if (action == GLFW_RELEASE)
    {
        switch (button)
        {
        case GLFW_MOUSE_BUTTON_LEFT:
            sceneSettings->SetMouseLeftClick(false);
            break;
        case GLFW_MOUSE_BUTTON_RIGHT:
            sceneSettings->SetMouseRightClick(false);
            break;
        }
    }
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }

    if (action == GLFW_PRESS)
    {
        switch (key)
        {
        case GLFW_KEY_Q: // A key on AZERTY
            /** ArcBall camera. */
            sceneSettings->SetCameraModel(CameraMovementModel::ARCBALL);
            break;
        case GLFW_KEY_F:
            /** FPS camera. */
            sceneSettings->SetCameraModel(CameraMovementModel::FPS);
            break;
        case GLFW_KEY_LEFT_SHIFT:
        case GLFW_KEY_RIGHT_SHIFT:
            sceneSettings->SetShiftKey(true);
            break;
        case GLFW_KEY_LEFT_CONTROL:
        case GLFW_KEY_RIGHT_CONTROL:
            sceneSettings->SetCtrlKey(true);
            break;
        case GLFW_KEY_LEFT_ALT:
        case GLFW_KEY_RIGHT_ALT:
            sceneSettings->SetAltKey(true);
            break;
        }
    }

    if (action == GLFW_RELEASE)
    {
        switch (key)
        {
        case GLFW_KEY_LEFT_SHIFT:
        case GLFW_KEY_RIGHT_SHIFT:
            sceneSettings->SetShiftKey(false);
            break;
        case GLFW_KEY_LEFT_CONTROL:
        case GLFW_KEY_RIGHT_CONTROL:
            sceneSettings->SetCtrlKey(false);
            break;
        case GLFW_KEY_LEFT_ALT:
        case GLFW_KEY_RIGHT_ALT:
            sceneSettings->SetAltKey(false);
            break;
        }
    }
}

void Statistics()
{
    GLfloat lineWidthRange[2] = {0.0f, 0.0f};
    glGetFloatv(GL_ALIASED_LINE_WIDTH_RANGE, lineWidthRange);
    std::cout << "Max line width supported: " << lineWidthRange[1] << std::endl;
}

GLFWwindow *GLFWInitialization()
{

    glfwSetErrorCallback(error_callback);

    if (!glfwInit())
    {
        std::cerr << "Error in glfw init function." << std::endl;
        exit(EXIT_FAILURE);
    }

    /** Anti-aliasing */
    glfwWindowHint(GLFW_SAMPLES, 16);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_SRGB_CAPABLE, GL_TRUE);

    /** Create and get the window's pointer. */
    GLFWwindow *window = glfwCreateWindow(sceneSettings->GetViewportWidth(), sceneSettings->GetViewportHeight(), "DRTMCVFX 3D", NULL, NULL);

    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(-1);
    }

    glfwMakeContextCurrent(window);

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    // Hide the mouse and enable unlimited mouvement
    // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    glfwPollEvents();
    glfwSetCursorPos(window, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);

    glfwSetKeyCallback(window, key_callback);

    glfwSetMouseButtonCallback(window, mouse_button_callback);

    glfwSetScrollCallback(window, scroll_callback);

    return window;
}

void GLEWInitialization()
{
    /** Init Glew. **/
    glewExperimental = GL_TRUE;
    GLenum ret = glewInit();

    if (ret != GLEW_OK)
    {
        std::cerr << "Glew Init error: " << ret << std::endl;
        exit(1);
    }
}

void ImGUIInitialization(GLFWwindow *window)
{
    /** Setup Dear ImGui context */
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    /** Setup Platform/Renderer bindings */
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    const char *glsl_version = "#version 330";
    ImGui_ImplOpenGL3_Init(glsl_version);
    /** Setup Dear ImGui style */
    ImGui::StyleColorsDark();
}

void OpenCVInitialization()
{
    // VideoCapture cap(0);

    // if (!cap.isOpened())
    // {
    //     std::cout << "cannot open camera" << std::endl;
    // }
    // else
    // {
    //     std::cout << "camera opened!" << std::endl;
    //     Mat image ;
    //     while(true){
    //         cap >> image;
    //         imshow("Display window", image);
    //         waitKey(25);
    //     }
    // }
}

int main(void)
{
    Statistics();

    /** Initialize everything that matter GLFW. */
    GLFWwindow *window = GLFWInitialization();

    /** Create the camera object. */
    Camera camera(window, sceneSettings);

    /** Init GLEW. */
    GLEWInitialization();

    /** Init Dear ImGUI. */
    ImGUIInitialization(window);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);

    glEnable(GL_MULTISAMPLE);

    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    // Set colors used when calling clear.
    GLclampf red = 0.2f, green = 0.2f, blue = 0.2f, alpha = 0.2f;
    glClearColor(red, green, blue, alpha);

    // Cull triangles which normal is not towards the camera
    glEnable(GL_CULL_FACE);

    auto cubePipeline = std::make_shared<ShaderPipeline>("../src/shaders/v_shader.glsl", "../src/shaders/f_shader.glsl");
    UnitCube cube(cubePipeline);

    auto meshPipeline = std::make_shared<ShaderPipeline>("../src/shaders/v_mesh.glsl", "../src/shaders/f_mesh.glsl");
    // Model model(meshPipeline, "/home/hepiteau/Work/DRTMCVFX/data/sphere.obj");
    Model model(meshPipeline, "/home/hepiteau/Work/DRTMCVFX/shape-reconstructor/data/bust/marble_bust_01_4k.fbx");

    // auto skyboxPipeline = std::make_shared<ShaderPipeline>("../src/shaders/v_skybox.glsl", "../src/shaders/f_skybox.glsl");

    // std::vector<std::string> faces{
    //     "/home/hepiteau/Work/DRTMCVFX/shape-reconstructor/data/hdri_cm/1/px.jpg",
    //     "/home/hepiteau/Work/DRTMCVFX/shape-reconstructor/data/hdri_cm/1/nx.jpg",
    //     "/home/hepiteau/Work/DRTMCVFX/shape-reconstructor/data/hdri_cm/1/py.jpg",
    //     "/home/hepiteau/Work/DRTMCVFX/shape-reconstructor/data/hdri_cm/1/ny.jpg",
    //     "/home/hepiteau/Work/DRTMCVFX/shape-reconstructor/data/hdri_cm/1/pz.jpg",
    //     "/home/hepiteau/Work/DRTMCVFX/shape-reconstructor/data/hdri_cm/1/nz.jpg"};

    // SkyBox skybox(skyboxPipeline, faces);

    // auto overlayPlanePipeline = std::make_shared<ShaderPipeline>("../src/shaders/v_overlay_plane.glsl", "../src/shaders/f_overlay_plane.glsl");
    // OverlayPlane overlayPlane(overlayPlanePipeline);
    // CudaTexture cudaTex(1080, 720);

    // auto vertices = std::vector<glm::vec3>{
    //     glm::vec3(0.0, 0.0, 0.0),
    //     glm::vec3(1.0, 1.0, 0.0),
    //     glm::vec3(1.0, 1.0, 0.0),
    //     glm::vec3(2.0, 1.0, 0.0)
    // };
    glm::mat4 ext = camera.GetViewMatrix();

    Utils::print(ext);

    glm::mat3 R = glm::mat3(ext);

    glm::vec3 center = glm::vec3(ext[0][3], ext[1][3], ext[2][3]);
    glm::vec3 pos = camera.GetPosition();

    std::cout << "Center: " << std::endl;
    Utils::print(center);
    std::cout << "Pos: " << std::endl;
    Utils::print(pos);

    glm::mat3x3 RT = glm::transpose(R);

    glm::vec3 cameraCoords = glm::vec3(-1.0 * sceneSettings->GetViewportRatio(), 1.0, -1.0);

    glm::vec3 res = RT * cameraCoords - RT * center + pos;

    // glm::vec4 res1 = Projection::CameraToWorld(glm::vec4(cameraCoords, 1.0f), ext, pos);

    int width = 1080/32;
    int height = 720/32;
    double wres = 1.0 / (double)width;
    double hres = 1.0 / (double)height;

    glm::vec3 wwMin = glm::vec3(-1.0 * sceneSettings->GetViewportRatio(), -1.0, -1.0);
    glm::vec3 wwMax = glm::vec3(1.0 * sceneSettings->GetViewportRatio(), 1.0, -1.0);
    glm::vec3 wwDelta = wwMax - wwMin;
    glm::vec3 wwRes = wwDelta / (float)width;
    glm::vec3 wwResD2 = wwDelta / 2.0f;

    glm::vec4 res1 = Projection::CameraToWorld(glm::vec4(wwMax.x, 0.0, -1.0, 1.0f), ext, pos);

    glm::vec3 adj = glm::vec3(0.0, 0.0, 1.0);
    glm::vec3 hypo = glm::vec3(pos.x - res1.x, pos.y - res1.y, pos.z - res1.z);

    std::cout << "Angle: (FOV):  " << glm::length(adj) << " / " << glm::length(hypo) << " | " << (acos(glm::length(adj) /glm::length(hypo) )) << std::endl;

    float *vertices2 = new float[6]{
        pos.x, pos.y, pos.z,
        res1.x, res1.y, res1.z

    };
    // float *vertices2 = new float[6 * width * height]{0.0f};

    // int cpt = 0;
    // for (int j = 0; j < height; ++j)
    // {
    //     for (int i = 0; i < width; ++i)
    //     {
    //         WRITE_VEC3(vertices2, cpt, pos);
    //         cpt += 3;
    //         glm::vec3 tmp = glm::vec3((-1.0 + i * 2 * wres + wres) * sceneSettings->GetViewportRatio(), -1.0 + j * 2 * hres, -1.0);
    //         glm::vec4 res1 = Projection::CameraToWorld(glm::vec4(tmp, 1.0f), ext, pos);
    //         WRITE_VEC3(vertices2, cpt, res1);
    //         cpt += 3;
    //     }
    // }

    std::cout << "Initialize rays. " << std::endl;

    Lines testLines(vertices2, 6 );
    // Lines testLines(vertices2, 6 * width * height);

    Lines cameraLines(camera.GetWireframe(), 16 * 3);
    cameraLines.SetColor(1.0, 0.0, 0.0, 0.5);

    Gizmo cameraGizmo(camera.GetPosition(), camera.GetRight(), camera.GetRealUp(), camera.GetForward());

    Volume3D volume;

    LineGrid lineGrid;

    // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // cudaTex.RunCUDA();

    Texture2D testImage("/home/hpiteau/work/shape-reconstructor/fiducial.png");

    static float scale = 1.0f;
    while (!glfwWindowShouldClose(window))
    {
        pxl_glfw_fps(window);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        /** ImGUI */
        // feed inputs to dear imgui, start new frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        /** MVP */
        camera.ComputeMatricesFromInputs();
        glm::mat4 projectionMatrix = camera.GetProjectionMatrix();
        glm::mat4 viewMatrix = camera.GetViewMatrix();

        // cube.Render(projectionMatrix, viewMatrix, camera.GetPosition(), WINDOW_WIDTH, WINDOW_HEIGHT);
        // skybox.Render(projectionMatrix, viewMatrix);

        model.Render(projectionMatrix, viewMatrix);

        testLines.Render(projectionMatrix, viewMatrix);
        cameraLines.Render(projectionMatrix, viewMatrix);
        cameraGizmo.Render(projectionMatrix, viewMatrix);

        // overlayPlane.Render(true, cudaTex.GetTex());

        volume.RenderWireframe(projectionMatrix, viewMatrix, sceneSettings);

        lineGrid.RenderWireframe(projectionMatrix, viewMatrix, sceneSettings);

        /** ImGUI */
        // render your GUI
        ImGui::Begin("Main Settings");
        ImGui::Button("Import Image Set");
        bool check = true;
        ImGui::Checkbox("Images uses same camera", &check);
        ImGui::Text(
            (std::string("Camera Mode: ") + std::string(sceneSettings->GetCameraModel() == CameraMovementModel::ARCBALL ? "ArcBall" : "Fps")).c_str());
        ImGui::Text(
            (std::string("Mouse Left: ") + std::string(sceneSettings->GetMouseLeftClick() ? "Pressed" : "Released")).c_str());
        ImGui::Text(
            (std::string("Mouse Right: ") + std::string(sceneSettings->GetMouseRightClick() ? "Pressed" : "Released")).c_str());
        ImGui::Text(
            (std::string("Scroll offsets: ") + std::to_string(sceneSettings->GetScrollOffsets().x) + std::string(", ") + std::to_string(sceneSettings->GetScrollOffsets().y)).c_str());

        // ImGui::Image((void*)(intptr_t)testImage.GetID(), ImVec2(testImage.GetWidth(), testImage.GetHeight()));
        ImGui::Separator();
        ImGui::Image((void *)(intptr_t)testImage.GetID(), ImVec2(256, 256), ImVec2(0, 0), ImVec2(1, 1));
        const char *items = "Image1\0Image2\0";
        int current = -1;
        // ImGui::Combo(items, &current);
        ImGui::Separator();
        ImGui::End();

        // Render dear imgui into screen
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);

    glfwTerminate();
    exit(EXIT_SUCCESS);
}