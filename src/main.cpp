#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <opencv2/opencv.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <tgmath.h>

#include <sstream>
#include "model/Camera/Camera.hpp"
#include "model/ShaderPipeline.hpp"
#include "model/CudaTexture.hpp"
#include "model/Volume3D.hpp"
#include "maths/MMath.hpp"
#include "utils/FileUtils.hpp"
#include "utils/Utils.hpp"
#include "view/Model.hpp"
#include "view/UnitCube.hpp"
#include "view/Grid.hpp"
#include "view/SkyBox.hpp"
#include "view/OverlayPlane.hpp"
#include "view/Plane.hpp"
#include "view/Lines.hpp"
#include "view/Gizmo.hpp"
#include "view/LineGrid.hpp"
#include "utils/SceneSettings.hpp"
#include "utils/Projection.h"

#include "controllers/Scene/Scene.hpp"

#include "controllers/AppController/AppController.hpp"

#include "../include/imgui/imgui.h"
#include "../include/imgui/backends/imgui_impl_glfw.h"
#include "../include/imgui/backends/imgui_impl_opengl3.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../../include/stb_image.h"
#include "../../include/stb_image_write.h"

#include "../../include/icons/IconsFontAwesome6.h"

#include "cuda/CudaSurface3D.cuh"
#include "utils/filesystem.h"

using namespace cv;
using namespace glm;

#define GLSL(src) #src
#define WINDOW_WIDTH 1920
#define WINDOW_HEIGHT 1080

/**
 * @brief
 */
std::shared_ptr<SceneSettings> sceneSettings = std::make_shared<SceneSettings>(WINDOW_WIDTH, WINDOW_HEIGHT);

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

        std::ostringstream oss;
        glfwGetFramebufferSize(window, &width, &height);

        oss << "(" << width << "," << height << ") - FPS: " << fps;

        glfwSetWindowTitle(window, oss.str().c_str());

        frame_count = 0;
    }

    frame_count++;
}

// static void render(GLFWwindow *window, GLuint shaderProgram)
// {
//     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_ACCUM_BUFFER_BIT);
// }

static void error_callback(int error, const char *description)
{
    fputs(description, stderr);
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    // don't pass mouse and keyboard presses further if an ImGui widget is active
    auto &io = ImGui::GetIO();
    if (io.WantCaptureMouse || io.WantCaptureKeyboard)
    {
        return;
    }

    sceneSettings->Scroll(xoffset, yoffset);
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    // don't pass mouse and keyboard presses further if an ImGui widget is active
    auto &io = ImGui::GetIO();
    if (io.WantCaptureMouse || io.WantCaptureKeyboard)
    {
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
        case GLFW_KEY_8:
            sceneSettings->IncreaseScrollSpeed();
            break;
        case GLFW_KEY_2:
            sceneSettings->DecreaseScrollSpeed();
            break;
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
        case GLFW_KEY_V:
            /** Enable / Disable Volume Rendering. **/
            sceneSettings->SetVariable(SceneGlobalVariables::VOLUME_RENDERING, false);
            break;
        case GLFW_KEY_P:
            char path[64] = {"screen.png"};
            Utils::SaveImage(path, window, sceneSettings->GetViewportWidth(), sceneSettings->GetViewportHeight());
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

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);

    /** Icons */
    io.Fonts->AddFontDefault();
    float baseFontSize = 18.0f;                      // 13.0f is the size of the default font. Change to the font size you use.
    float iconFontSize = baseFontSize * 2.0f / 3.0f; // FontAwesome fonts need to have their sizes reduced by 2.0f/3.0f in order to align correctly

    // merge in icons from Font Awesome
    static const ImWchar icons_ranges[] = {ICON_MIN_FA, ICON_MAX_16_FA, 0};
    ImFontConfig icons_config;
    icons_config.MergeMode = true;
    icons_config.PixelSnapH = true;
    icons_config.GlyphMinAdvanceX = iconFontSize;
    io.Fonts->AddFontFromFileTTF("../include/icons/fa-solid-900.ttf", iconFontSize, &icons_config, icons_ranges);
    // use FONT_ICON_FILE_NAME_FAR if you want regular instead of solid
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

void GLInitialization()
{
    glEnable(GL_BLEND);
    glEnable(GL_TEXTURE_2D);
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
//    GLclampf red = 0.95f, green = 0.95f, blue = 0.95f, alpha = 1.0f;
    glClearColor(red, green, blue, alpha);

    // Cull triangles which normal is not towards the camera
    glEnable(GL_CULL_FACE);
}

void atexit_handler()
{
    std::cout << "atexit handler\n";
}


int main(void)
{

    const int result = std::atexit(atexit_handler); // handler will be called

    if (result != 0)
    {
        std::cerr << "atexit registration failed\n";
        return EXIT_FAILURE;
    }


    Statistics();

    /** Initialize everything that matter GLFW. */
    GLFWwindow *window = GLFWInitialization();

    /** Init GLEW. */
    GLEWInitialization();

    /** Init Dear ImGUI. */
    ImGUIInitialization(window);

    /** Set gl variables. */
    GLInitialization();

    /** Create the camera object. */
    std::shared_ptr<AppController> app = std::make_shared<AppController>(window, sceneSettings);

    Scene *scene = app->GetScene();

    auto cubePipeline = std::make_shared<ShaderPipeline>("../src/shaders/v_shader.glsl", "../src/shaders/f_shader.glsl");
    UnitCube cube(cubePipeline);

    // auto meshPipeline = std::make_shared<ShaderPipeline>("../src/shaders/v_mesh.glsl", "../src/shaders/f_mesh.glsl");
    // Model model(meshPipeline, "/home/hepiteau/Work/DRTMCVFX/data/sphere.obj");
    // Model model(scene, meshPipeline, "/home/hepiteau/Work/DRTMCVFX/shape-reconstructor/data/bust/marble_bust_01_4k.fbx");

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
    
    scene->GetActiveCam()->UpdateWireframe();
    Lines cameraLines(scene, scene->GetActiveCam()->GetWireframe(), 16 * 3);
    cameraLines.SetColor(1.0, 0.0, 0.0, 0.5);


    // Image image = Image("../screen1.png");
    // image.LoadPng("../data/nerf/train/r_0.png", true, false);

    // Texture2D testImageTex = Texture2D();
    // testImageTex.LoadFromImage(&image);

    // Plane m_imagePlane = Plane(scene);
    // m_imagePlane.SetTexture2D(&testImageTex);

    // Volume3D volume;

    // LineGrid lineGrid;

    // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // cudaTex.RunCUDA();

    // CudaSurface3D surface(100, 100, 100);

    // static float scale = 1.0f;
    cudaSetDevice(0);
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
        scene->GetActiveCam()->ComputeMatricesFromInputs(window);
        
        // cube.Render(projectionMatrix, viewMatrix, camera.GetPosition(), WINDOW_WIDTH, WINDOW_HEIGHT);
        // skybox.Render(projectionMatrix, viewMatrix);
        // model.Render(projectionMatrix, viewMatrix, sceneSettings);
        // testLines.Render(projectionMatrix, viewMatrix, sceneSettings);

        app->Render();
        // m_imagePlane.Render();

        // lineGrid.Render(projectionMatrix, viewMatrix, sceneSettings);

        // auto started = std::chrono::high_resolution_clock::now();
        // auto done = std::chrono::high_resolution_clock::now();
        // std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count() << std::endl;

        // overlayPlane.Render(true, cudaTex.GetTex());

        // volume.Render(projectionMatrix, viewMatrix, sceneSettings);



        ImGui::Begin("Main Settings");
        if (ImGui::BeginMainMenuBar())
        {
            if (ImGui::BeginMenu("File"))
            {
                ImGui::MenuItem("Open calibration images");
                ImGui::Separator();
                static bool vsync = true;
                if(ImGui::MenuItem("Enable v-sync.", NULL, &vsync)){
                    glfwSwapInterval(vsync);
                }

                ImGui::Separator();
                ImGui::MenuItem(ICON_FA_SQUARE_XMARK " Exit", "Esc");
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Setups"))
            {
                ImGui::MenuItem("Setup for Simple Stereo");
                ImGui::MenuItem("Setup for MultiView");
                ImGui::EndMenu();
            }

            ImGui::EndMainMenuBar();
        }

        ImGui::SeparatorText("Memory Usage: ");
        
        size_t free = 0, total = 0;
        // cudaError_t err = 
        cudaMemGetInfo(&free,  &total);
        free = free/1024/1024;
        total = total/1024/1024;
        
        ImGui::Text("Free: ");
        ImGui::SameLine();
        ImGui::TextUnformatted(std::to_string(free).c_str());
        ImGui::SameLine();
        ImGui::Text(" MB");
        
        ImGui::Text("Total: ");
        ImGui::SameLine();
        ImGui::TextUnformatted(std::to_string(total).c_str());
        ImGui::SameLine();
        ImGui::Text(" MB");

        ImGui::Text("Used: ");
        ImGui::SameLine();
        ImGui::TextUnformatted(std::to_string(total - free).c_str());
        ImGui::SameLine();
        ImGui::Text(" MB");

        ImGui::Separator();

        ImGui::TextUnformatted(
            (std::string("Camera Mode: ") + std::string(sceneSettings->GetCameraModel() == CameraMovementModel::ARCBALL ? "ArcBall" : "Fps")).c_str());
        ImGui::TextUnformatted(
            (std::string("Mouse Left: ") + std::string(sceneSettings->GetMouseLeftClick() ? "Pressed" : "Released")).c_str());
        ImGui::TextUnformatted(
            (std::string("Mouse Right: ") + std::string(sceneSettings->GetMouseRightClick() ? "Pressed" : "Released")).c_str());
        ImGui::TextUnformatted(
            (std::string("Scroll offsets: ") + std::to_string(sceneSettings->GetScrollOffsets().x) + std::string(", ") + std::to_string(sceneSettings->GetScrollOffsets().y)).c_str());

        float inf[3] = {scene->GetActiveCam()->GetPosition().x, scene->GetActiveCam()->GetPosition().y, scene->GetActiveCam()->GetPosition().z};
        ImGui::InputFloat3("Camera position", inf);
        if(inf[0] != scene->GetActiveCam()->GetPosition()[0]
        || inf[1] != scene->GetActiveCam()->GetPosition()[1]
        || inf[2] != scene->GetActiveCam()->GetPosition()[2]){
            scene->GetActiveCam()->SetPosition(inf[0], inf[1], inf[2]);
        }

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