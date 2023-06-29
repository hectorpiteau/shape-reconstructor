/*
Author: Hector Piteau (hector.piteau@gmail.com)
f_plane.glsl (c) 2023
Desc: Plane shader with uv coords.
Created:  2023-04-19T09:27:31.466Z
Modified: 2023-04-20T09:53:10.796Z
*/

#version 330 core

uniform usampler2D texture0;
uniform ivec2 mousePos;

in vec2 TexCoord;

out vec4 FragColor;

layout(origin_upper_left, pixel_center_integer) in vec4 gl_FragCoord;

vec4 mouseData;

void main()
{
    FragColor = texture(texture0, TexCoord) / 255.0f;

    if(mousePos == ivec2(gl_FragCoord)) mouseData = FragColor;
}