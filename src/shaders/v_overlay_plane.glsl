#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

uniform float scale;

out vec3 normal;
out vec4 FragPos;
out vec2 TexCoord;

void main()
{
    gl_Position = vec4(scale * aPos.x, aPos.y, scale* aPos.z, 1.0);
    TexCoord = aTexCoord;
}
