/*
Author: Hector Piteau (hector.piteau@gmail.com)
v_plane.glsl (c) 2023
Desc: PLane shader with uv coords.
Created:  2023-04-19T09:28:04.221Z
Modified: 2023-04-20T09:22:12.113Z
*/

#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec4 FragPos;
out vec2 TexCoord;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}