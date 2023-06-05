#version 330 core


// uniform sampler2D texture0;
uniform usampler2D texture0;

in vec2 TexCoord;

out vec4 FragColor;


void main()
{
    FragColor = vec4(0.0, 0.0, 0.0, 1.0);

    vec4 tex_res = texture(texture0, TexCoord) / 255.0;
    // FragColor.r = FragColor.r / 255.0;
    // if( tex_res.r == 0.1){
    //     FragColor.r = 1.0;
    // }else{
    //     FragColor.r = tex_res.r / 255.0;
    // }
    FragColor = tex_res;
    // FragColor.r = tex_res.r;
    // FragColor.g = tex_res.g;
    // FragColor.b = tex_res.b;// + 0.1;
    // FragColor.w = tex_res.a;// * 1.0;
    // FragColor = vec4(1.0, 0.0, 1.0, 0.8);
}