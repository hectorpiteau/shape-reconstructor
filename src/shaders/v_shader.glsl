#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 model;   // Model matrix
uniform mat4 view;    // View matrix
uniform mat4 projection; // Projection matrix

out vec3 Normal;
out vec3 FragPos;

void main()
{
    // gl_Position = mvp * vec4(scale*0.001 + aPos.x, 0.5 + aPos.y, aPos.z, 1.0);

    vec4 worldPos = model * vec4(aPos, 1.0);
    FragPos = vec3(worldPos);
    // Normal = mat3(transpose(inverse(model))) * aNormal;
    Normal = aNormal;

    gl_Position = projection * view * worldPos;
}
