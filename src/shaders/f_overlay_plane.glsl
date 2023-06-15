#version 330 core


// uniform sampler2D texture0;
uniform usampler2D texture0;

in vec2 TexCoord;
out vec4 FragColor;


void main()
{
    FragColor = vec4(0.0, 0.0, 0.0, 1.0);

    vec4 tex_res = texture(texture0, TexCoord) / 255.0;

    FragColor = tex_res;
}