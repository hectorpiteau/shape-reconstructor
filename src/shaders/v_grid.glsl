#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

// uniform float view;
// uniform float proj;
uniform float scale;
uniform mat4 mvp;

out vec3 normal;
out vec4 FragPos;

void main()
{
    gl_Position = mvp * vec4(scale * aPos.x, aPos.y, scale* aPos.z, 1.0);
    FragPos = vec4(scale * aPos.x, aPos.y, scale* aPos.z, 1.0);
    normal = aNormal;
}
