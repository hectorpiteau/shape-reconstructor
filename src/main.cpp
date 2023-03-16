#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

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

using namespace cv;

using namespace glm;

#define GLSL(src) #src
#define WINDOW_WIDTH 1080
#define WINDOW_HEIGHT 720

float FOV = 90.0f;
float NEAR = 1.0f;
float FAR = 10.0f;

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

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

int main(void)
{
    struct ScreenInfos screenInfos = {.width = WINDOW_WIDTH, .height = WINDOW_HEIGHT};

    // float dimensions[2] = {WINDOW_WIDTH, WINDOW_HEIGHT};

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
    GLFWwindow *window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "DRTMCVFX 3D", NULL, NULL);

    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    /** Create the camera object. */
    Camera camera(window, screenInfos);

    glfwMakeContextCurrent(window);

    /** Init Glew. **/
    glewExperimental = GL_TRUE;
    GLenum ret = glewInit();

    if (ret != GLEW_OK)
    {
        std::cerr << "Glew Init error: " << ret << std::endl;
        return 1;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    // Hide the mouse and enable unlimited mouvement
    // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    glfwPollEvents();
    glfwSetCursorPos(window, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);

    glfwSetKeyCallback(window, key_callback);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);

    glEnable(GL_MULTISAMPLE);

    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT,  GL_NICEST);

    // Set colors used when calling clear.
    GLclampf red = 0.2f, green = 0.2f, blue = 0.2f, alpha = 0.2f;
    glClearColor(red, green, blue, alpha);

    // Cull triangles which normal is not towards the camera
    glEnable(GL_CULL_FACE);

    auto cubePipeline = std::make_shared<ShaderPipeline>("../src/shaders/v_shader.glsl", "../src/shaders/f_shader.glsl");
    UnitCube cube(cubePipeline);

    auto gridPipeline = std::make_shared<ShaderPipeline>("../src/shaders/v_grid.glsl", "../src/shaders/f_grid.glsl");
    Grid grid(gridPipeline);

    // auto meshPipeline = std::make_shared<ShaderPipeline>("../src/shaders/v_mesh.glsl", "../src/shaders/f_mesh.glsl");
    // Model model(meshPipeline, "/home/hepiteau/Work/DRTMCVFX/data/sphere.obj");
    // Model model(meshPipeline, "/home/hepiteau/Work/DRTMCVFX/data/bust/marble_bust_01_4k.fbx");

    // auto skyboxPipeline = std::make_shared<ShaderPipeline>("../src/shaders/v_skybox.glsl", "../src/shaders/f_skybox.glsl");

    // std::vector<std::string> faces{
    //     "/home/hepiteau/Work/DRTMCVFX/data/hdri_cm/1/px.jpg",
    //     "/home/hepiteau/Work/DRTMCVFX/data/hdri_cm/1/nx.jpg",
    //     "/home/hepiteau/Work/DRTMCVFX/data/hdri_cm/1/py.jpg",
    //     "/home/hepiteau/Work/DRTMCVFX/data/hdri_cm/1/ny.jpg",
    //     "/home/hepiteau/Work/DRTMCVFX/data/hdri_cm/1/pz.jpg",
    //     "/home/hepiteau/Work/DRTMCVFX/data/hdri_cm/1/nz.jpg"};

    // SkyBox skybox(skyboxPipeline, faces);

    auto overlayPlanePipeline = std::make_shared<ShaderPipeline>("../src/shaders/v_overlay_plane.glsl", "../src/shaders/f_overlay_plane.glsl");
    OverlayPlane overlayPlane(overlayPlanePipeline);
    CudaTexture cudaTex(1080, 720);

    
    // auto vertices = std::vector<glm::vec3>{
    //     glm::vec3(0.0, 0.0, 0.0), 
    //     glm::vec3(1.0, 1.0, 0.0),
    //     glm::vec3(1.0, 1.0, 0.0),
    //     glm::vec3(2.0, 1.0, 0.0)
    // };

    float vertices2[12] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
        2.0f, 1.0f, 0.0f
    };

    Lines testLines(vertices2, 12);

    // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    cudaTex.RunCUDA();

    static float scale = 1.0f;
    while (!glfwWindowShouldClose(window))
    {
        pxl_glfw_fps(window);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        /** MVP */
        camera.ComputeMatricesFromInputs();
        glm::mat4 projectionMatrix = camera.GetProjectionMatrix();
        glm::mat4 viewMatrix = camera.GetViewMatrix();

        // cube.Render(projectionMatrix, viewMatrix, camera.GetPosition(), WINDOW_WIDTH, WINDOW_HEIGHT);
        // skybox.Render(projectionMatrix, viewMatrix);

        grid.Render(projectionMatrix, viewMatrix, WINDOW_WIDTH, WINDOW_HEIGHT);

        // model.Render(projectionMatrix, viewMatrix);
        
        testLines.Render(projectionMatrix, viewMatrix);

        // overlayPlane.Render(true, cudaTex.GetTex());

        // cudaTex.Render(overlayPlanePipeline);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);

    glfwTerminate();
    exit(EXIT_SUCCESS);
}