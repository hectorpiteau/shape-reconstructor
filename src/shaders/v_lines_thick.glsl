#version 330 core

layout(location = 0) in vec3 aPos;

out vec3 fragPos;

out DATA
{
    mat4 projection;
} data_out;

uniform mat4 mvp;

void main()
{
    gl_Position =  vec4(aPos, 1.0);
    fragPos = aPos;
    data_out.projection = mvp;
}

