/*
Author: Hector Piteau (hector.piteau@gmail.com)
f_plane.glsl (c) 2023
Desc: Plane shader with uv coords.
Created:  2023-04-19T09:27:31.466Z
Modified: 2023-04-20T09:53:10.796Z
*/

#version 330 core

// uniform usampler2D texture0;
uniform sampler2D texture0;

in vec2 TexCoord;
out vec4 FragColor;


void main()
{
    FragColor = texture(texture0, TexCoord);
}