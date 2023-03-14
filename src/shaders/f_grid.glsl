#version 330 core

out vec4 FragColor;
in vec4 gl_FragCoord;
in vec3 normal;

in vec4 FragPos;

uniform vec2 dims;

void main()
{
    vec2 st = gl_FragCoord.xy / dims;

    vec3 lightPos = vec3(1.2, 1.0, 2.0);
    vec3 objectColor = vec3(0.1, 0.0, 0.1);
    vec3 lightColor = vec3(1.0, 1.0, 1.0);

    float ambientStrength = 0.8;    
    vec3 ambient = ambientStrength * lightColor;

    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(lightPos - gl_FragCoord.xyz); 

    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    vec3 result = (ambient + diffuse) * objectColor;
    
        result.x = FragPos.x * 0.001 * norm.x + st.x *0.001;
        // result.y = FragPos.x + 0.001 * st.x;
        
        FragColor = vec4(result, 0.8);
        
        
        if (FragPos.x < -1.98) FragColor.xyz =  vec3(0.9,0.9,1.0);
        else if (FragPos.z < -1.98) FragColor.xyz =  vec3(0.9,0.9,1.0);
        else if (FragPos.x > 1.98) FragColor.xyz =  vec3(0.9,0.9,1.0);
        else if (FragPos.z > 1.98) FragColor.xyz =  vec3(0.9,0.9,1.0);
        else if (mod(FragPos.x, 0.5 ) < 0.005) FragColor.xyz =  vec3(0.6,0.6,0.6);
        else if (mod(FragPos.z, 0.5 ) < 0.005) FragColor.xyz =  vec3(0.6,0.6,0.6);
        // else if (mod(FragPos.x, 0.1 ) < 0.002) FragColor.z =  1.0;
        // else if (mod(FragPos.z, 0.1 ) < 0.002) FragColor.z =  1.0;
        else FragColor.w = 0.1;
        // if (FragPos.z - 1.0 < 0.002) FragColor.y =  1.0;
   
}