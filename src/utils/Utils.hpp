#pragma once

#include <GL/glew.h>
#include <stdlib.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>

#include "../../include/stb_image.h"
#include "../../include/stb_image_write.h"

#define WRITE_VEC3(array, index, vec3) \
    (array)[(index)] = (vec3).x;       \
    (array)[((index) + 1)] = (vec3).y; \
    (array)[((index) + 2)] = (vec3).z;

struct ScreenInfos
{
    int width;
    int height;
};

class Utils
{
public:
    template <typename T>
    static int Sign(T val)
    {
        return (T(0) < val) - (val < T(0));
    }

    static void print(const glm::mat4 &mat)
    {
        std::cout.precision(5);
        for (int i = 0; i < 4; ++i)
            std::cout << std::fixed << mat[0][i] << ",\t" << mat[1][i] << ",\t" << mat[2][i] << ",\t" << mat[3][i] << std::endl;
    }

    static void print(const glm::vec4 &vec)
    {
        std::cout.precision(5);
        std::cout << std::fixed << "(" << vec[0] << ",\t" << vec[1] << ",\t" << vec[2] << ",\t" << vec[3] << ")" << std::endl;
    }

    static void print(const glm::vec3 &vec)
    {
        std::cout.precision(5);
        std::cout << std::fixed << "(" << vec[0] << ",\t" << vec[1] << ",\t" << vec[2] << ")" << std::endl;
    }

    static void print(const glm::vec2 &vec)
    {
        std::cout.precision(5);
        std::cout << std::fixed << "(" << vec[0] << ",\t" << vec[1] << ")" << std::endl;
    }

    static unsigned int CreateVertexBuffer(const float *vertices)
    {
        unsigned int VBO;
        glEnable(GL_CULL_FACE);
        // glFrontFace(GL_CCW);
        glGenBuffers(1, &VBO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        return VBO;
    }

    static unsigned int CreateObjectBuffer()
    {
        unsigned int VAO;
        glGenVertexArrays(1, &VAO);
        glBindVertexArray(VAO);
        return VAO;
    }

    static GLenum LoadEXRFile(const std::string &path)
    {
        // try
        // {
        //     Imf::RgbaInputFile file(path.c_str());
        //     Imath::Box2i dw = file.dataWindow();
        //     int width = dw.max.x - dw.min.x + 1;
        //     int height = dw.max.y - dw.min.y + 1;

        //     Imf::Array2D<Imf::Rgba> pixels(width, height);

        //     file.setFrameBuffer(&pixels[0][0], 1, width);
        //     file.readPixels(dw.min.y, dw.max.y);
        // }
        // catch (const std::exception &e)
        // {
        //     std::cerr << "[Utils::LoadEXRFile] Error reading image file hello.exr:" << e.what() << std::endl;
        //     return 0;
        // }

        GLenum target = GL_TEXTURE_RECTANGLE_NV;
        // glGenTextures(2, imageTexture);
        // glBindTexture(target, imageTexture);
        // glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        // glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        // glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        // glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        // glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        // glTexImage2D(target, 0, GL_FLOAT_RGBA16_NV, dim.x, dim.y, 0, GL_RGBA, GL_HALF_FLOAT_NV, pixelBuffer);
        // glActiveTextureARB(GL_TEXTURE0_ARB);
        // glBindTexture(target, imageTexture);

        // glBegin(GL_QUADS);
        // glTexCoord2f(0.0, 0.0);
        // glVertex2f(0.0, 0.0);
        // glTexCoord2f(dim.x, 0.0);
        // glVertex2f(dim.x, 0.0);
        // glTexCoord2f(dim.x, dim.y);
        // glVertex2f(dim.x, dim.y);
        // glTexCoord2f(0.0, dim.y);
        // glVertex2f(0.0, dim.y);
        // glEnd();

        return target;
    }

    static void SaveImage(char *filepath, GLFWwindow *w, int width, int height)
    {
        glfwGetFramebufferSize(w, &width, &height);
        GLsizei nrChannels = 3;
        GLsizei stride = nrChannels * width;
        stride += (stride % 4) ? (4 - stride % 4) : 0;
        GLsizei bufferSize = stride * height;
        std::vector<char> buffer(bufferSize);
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        glReadBuffer(GL_FRONT);
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer.data());
        stbi_flip_vertically_on_write(true);
        stbi_write_png(filepath, width, height, nrChannels, buffer.data(), stride);
    }
};