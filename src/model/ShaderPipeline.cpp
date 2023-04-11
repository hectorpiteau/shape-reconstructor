#include <GL/glew.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "../utils/FileUtils.hpp"
#include "ShaderPipeline.hpp"

ShaderPipeline::ShaderPipeline(const std::string& vertexShaderFilename, const std::string& fragmentShaderFilename)
: m_vertexShaderFilename(vertexShaderFilename), m_fragmentShaderFilename(fragmentShaderFilename), m_programShader(0)
{
    CompileShader();
}

ShaderPipeline::~ShaderPipeline()
{}

void ShaderPipeline::CompileShader()
{
    /** Temporary containers used to store the stringified shaders. */
    std::string vertexShader, fragmentShader;

    GLenum err = glGetError();
    if (err != GL_NO_ERROR)
    {
        fprintf(stderr, "OpenGL error (at line scene.cpp:%d): %s\n", __LINE__, gluErrorString(err));
    }

    GLuint shaderProgram = glCreateProgram();

    if (shaderProgram == 0)
    {
        std::cerr << "ShaderPipeline: Error creating the shader." << std::endl;
        exit(1);
    }

    if (!FileUtils::ReadFile(m_vertexShaderFilename.c_str(), vertexShader))
    {
        std::cerr << "ShaderPipeline: Error reading the vertex file." << std::endl;
        exit(1);
    }

    GLuint vertexShaderID = AddShader(shaderProgram, vertexShader.c_str(), m_vertexShaderFilename, GL_VERTEX_SHADER);

    if (!FileUtils::ReadFile(m_fragmentShaderFilename.c_str(), fragmentShader))
    {
        std::cerr << "ShaderPipeline: Error reading the fragment file." << std::endl;
        exit(1);
    }

    GLuint fragmentShaderID = AddShader(shaderProgram, fragmentShader.c_str(), m_fragmentShaderFilename, GL_FRAGMENT_SHADER);

    GLint success = 0;
    GLchar errorLog[1024] = {0};

    glLinkProgram(shaderProgram);

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);

    if (success == 0)
    {
        glGetProgramInfoLog(shaderProgram, sizeof(errorLog), NULL, errorLog);
        std::cerr << "Error linking shader program: " << errorLog << std::endl;
        exit(1);
    }

    glValidateProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_VALIDATE_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(shaderProgram, sizeof(errorLog), NULL, errorLog);
        std::cerr << "Invalid shader program: " << errorLog << std::endl;
        exit(1);
    }
    else
    {
        std::cout << "ShaderPipeline: Success loading shader: " << m_vertexShaderFilename << std::endl;
        std::cout << "ShaderPipeline: Success loading shader: " << m_fragmentShaderFilename << std::endl
                  << "===" << std::endl;
    }

    glDetachShader(shaderProgram, vertexShaderID);
    glDetachShader(shaderProgram, fragmentShaderID);

    glDeleteShader(vertexShaderID);
    glDeleteShader(fragmentShaderID);

    glUseProgram(shaderProgram);

    /** Save the program shader's id for future use. */
    m_programShader = shaderProgram;
}

GLint ShaderPipeline::AddUniform(std::string name)
{
    GLint res = glGetUniformLocation(m_programShader, name.c_str());
    if (res == -1)
    {
        std::cerr << "Error creating uniform location: " << res << std::endl;
        exit(1);
    }
    m_uniforms[name] = res;
    return res;
}

GLint ShaderPipeline::GetUniform(std::string name)
{
    return m_uniforms[name];
}

void ShaderPipeline::UpdateShader()
{
    CompileShader();
}

void ShaderPipeline::UseShader()
{
    glUseProgram(m_programShader);
}

GLuint ShaderPipeline::AddShader(GLuint shaderProgram, const char *shader_str, std::string filename, GLenum shaderType)
{
    GLuint shaderObj = glCreateShader(shaderType);

    if (shaderObj == 0)
    {
        std::cerr << "Error creating a shader type: " << shaderType << std::endl;
        exit(1);
    }

    const GLchar *p[1] = {shader_str};

    GLint lengths[1] = {(int)strlen(shader_str)};

    glShaderSource(shaderObj, 1, p, lengths);
    glCompileShader(shaderObj);

    GLint success;
    glGetShaderiv(shaderObj, GL_COMPILE_STATUS, &success);

    if (!success)
    {
        GLchar infoLog[1024];
        glGetShaderInfoLog(shaderObj, 1024, NULL, infoLog);
        std::cerr << "Error compiling shader type: " << shaderType << " File: " << filename << " Error: " << std::endl
                  << infoLog << std::endl;
        exit(1);
    }

    glAttachShader(shaderProgram, shaderObj);

    return shaderObj;
}