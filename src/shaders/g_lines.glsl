#version 330 core

layout (lines) in;
layout(triangle_strip, max_vertices = 24) out;

in vec3 fragPos[];

//out vec3 Color;

in DATA
{
    mat4 projection;
} data_in[];

void main(){
//    vec3 lineVec = gl_in[1].gl_Position.xyz - gl_in[0].gl_Position.xyz;
//    vec3 normal = normalize(cross(lineVec, vec3(0.0, 0.0, 1.0))); // Adjust normal direction as needed
//    float width = 0.1; // Width of the rectangle strip
//
//    vec4 offset = vec4(normal * width, 0.0);
//
//    vec4 vertices[4];
//    vertices[0] = gl_in[0].gl_Position + offset;
//    vertices[1] = gl_in[0].gl_Position - offset;
//    vertices[2] = gl_in[1].gl_Position + offset;
//    vertices[3] = gl_in[1].gl_Position - offset;
//
//    for (int i = 0; i < 4; ++i) {
//        gl_Position = data_in[0].projection * vertices[i];
////        texCoord = vec2(i == 1 || i == 3 ? 1.0 : 0.0, i > 1 ? 1.0 : 0.0);
//        EmitVertex();
//    }
//    EndPrimitive();


    vec3 lineVec = gl_in[1].gl_Position.xyz - gl_in[0].gl_Position.xyz;
    vec3 up = vec3(0.0, 1.0, 0.0); // This can be adjusted to fit the orientation needed
    if(lineVec == vec3(0.0, 1.0, 0.0) || lineVec == vec3(0.0, -1.0, 0.0)){
        up = vec3(0.0, 0.0, 1.0);
    }
    vec3 right = normalize(cross(up, lineVec)); // Right vector
    up = normalize(cross(lineVec, right)); // Correct up vector

    float width = 0.001; // Half width of the rectangle strip
    float height = 0.001; // Half height of the rectangle strip

    vec4 offset1 = vec4(right * width, 0.0);
    vec4 offset2 = vec4(up * height, 0.0);

    vec4 vertices[8];
    vertices[0] = gl_in[0].gl_Position + offset1 + offset2;
    vertices[1] = gl_in[0].gl_Position + offset1 - offset2;
    vertices[2] = gl_in[0].gl_Position - offset1 + offset2;
    vertices[3] = gl_in[0].gl_Position - offset1 - offset2;
    vertices[4] = gl_in[1].gl_Position + offset1 + offset2;
    vertices[5] = gl_in[1].gl_Position + offset1 - offset2;
    vertices[6] = gl_in[1].gl_Position - offset1 + offset2;
    vertices[7] = gl_in[1].gl_Position - offset1 - offset2;

    // Create 6 faces for the rectangular box

    // Front face
    gl_Position = data_in[0].projection * vertices[0];
//    texCoord = vec2(0.0, 0.0);
    EmitVertex();
    gl_Position = data_in[0].projection * vertices[2];
//    texCoord = vec2(1.0, 0.0);
    EmitVertex();
    gl_Position = data_in[0].projection * vertices[4];
//    texCoord = vec2(0.0, 1.0);
    EmitVertex();
    gl_Position = data_in[0].projection * vertices[6];
//    texCoord = vec2(1.0, 1.0);
    EmitVertex();
    EndPrimitive();

    // Back face
    gl_Position = data_in[0].projection * vertices[1];
//    texCoord = vec2(0.0, 0.0);
    EmitVertex();
    gl_Position = data_in[0].projection * vertices[3];
//    texCoord = vec2(1.0, 0.0);
    EmitVertex();
    gl_Position = data_in[0].projection * vertices[5];
//    texCoord = vec2(0.0, 1.0);
    EmitVertex();
    gl_Position = data_in[0].projection * vertices[7];
//    texCoord = vec2(1.0, 1.0);
    EmitVertex();
    EndPrimitive();

    // Left face
    gl_Position = data_in[0].projection * vertices[2];
//    texCoord = vec2(0.0, 0.0);
    EmitVertex();
    gl_Position = data_in[0].projection * vertices[3];
//    texCoord = vec2(1.0, 0.0);
    EmitVertex();
    gl_Position = data_in[0].projection * vertices[6];
//    texCoord = vec2(0.0, 1.0);
    EmitVertex();
    gl_Position = data_in[0].projection * vertices[7];
//    texCoord = vec2(1.0, 1.0);
    EmitVertex();
    EndPrimitive();

    // Right face
    gl_Position = data_in[0].projection * vertices[0];
//    texCoord = vec2(0.0, 0.0);
    EmitVertex();
    gl_Position = data_in[0].projection * vertices[1];
//    texCoord = vec2(1.0, 0.0);
    EmitVertex();
    gl_Position = data_in[0].projection * vertices[4];
//    texCoord = vec2(0.0, 1.0);
    EmitVertex();
    gl_Position = data_in[0].projection * vertices[5];
//    texCoord = vec2(1.0, 1.0);
    EmitVertex();
    EndPrimitive();

    // Top face
    gl_Position = data_in[0].projection * vertices[0];
//    texCoord = vec2(0.0, 0.0);
    EmitVertex();
    gl_Position = data_in[0].projection * vertices[2];
//    texCoord = vec2(1.0, 0.0);
    EmitVertex();
    gl_Position = data_in[0].projection * vertices[1];
//    texCoord = vec2(0.0, 1.0);
    EmitVertex();
    gl_Position = data_in[0].projection * vertices[3];

    EmitVertex();
    EndPrimitive();

    // Bottom face
    gl_Position = data_in[0].projection * vertices[4];
    EmitVertex();
    gl_Position = data_in[0].projection * vertices[6];
    EmitVertex();
    gl_Position = data_in[0].projection * vertices[5];
    EmitVertex();
    gl_Position = data_in[0].projection * vertices[7];
    EmitVertex();
    EndPrimitive();


}