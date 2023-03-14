#ifndef SHADER_PIPELINE_H
#define SHADER_PIPELINE_H
#include <string>
#include <GL/gl.h>
#include <map>

class ShaderPipeline {
public:
    /**
     * @brief Construct a new Shader Pipeline object from vertex filenames.
     * 
     * @param vertexShaderFilename : The path to the vertex shader's file. 
     * @param fragmentShaderFilename : The path to the fragment shader's file.
     */
    ShaderPipeline(std::string vertexShaderFilename, std::string fragmentShaderFilename);

    /** Delete copy constructor. */
    ShaderPipeline(const ShaderPipeline&) = delete;

    ~ShaderPipeline();

    
    /**
     * @brief Compile the shaders. Read files and load both shaders into the local program.
     */
    void CompileShader();
    
    /**
     * @brief Re-compile shaders from the save files.
     */
    void UpdateShader();

    /**
     * @brief Call glUseProgram on the pipeline's shaders.
     */
    void UseShader();

    GLint AddUniform(std::string name);
    GLint GetUniform(std::string name);

    void SetFloat(const std::string &name, float value) const
    { 
        glUniform1f(glGetUniformLocation(m_programShader, name.c_str()), value); 
    }

    GLuint m_programShader;
private:
    std::string m_vertexShaderFilename;
    std::string m_fragmentShaderFilename;
    std::map<std::string, GLint> m_uniforms; 

    GLuint AddShader(GLuint shaderProgram, const char *shader_str,  std::string filename, GLenum shaderType);
};

#endif // SHADER_PIPELINE_H