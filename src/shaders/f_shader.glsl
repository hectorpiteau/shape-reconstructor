#version 330 core

out vec4 FragColor;


in vec3 Normal;
in vec3 FragPos;


uniform vec3 viewPos;   // Position of the camera

void main()
{
    vec3 norm = normalize(Normal);
    // FragColor = vec4(FragPos.x, viewPos.y, Normal.z, 1.0);
    

    vec3 lightPos = vec3(1.2, 1.0, 2.0);
    // vec3 lightPos = vec3(0.0, 0.0, 3.0);
    vec3 objectColor = vec3(0.8, 0.2, 0.8);
    vec3 lightColor = vec3(1.0, 0.9, 0.9);

    /** Ambient lightning */
    float ambientStrength = 0.7;    
    vec3 ambient = ambientStrength * lightColor;

    /** Diffuse lightning */
    vec3 lightDir = normalize(lightPos - FragPos); 
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    /** Specular lightning */
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, Normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);
    
    vec3 result = (ambient + diffuse + specular) * objectColor;
    
    // if(FragPos.x < -0.49){
    //     FragColor = vec4(1.0, 1.0, 1.0, 1.0);    
    // }else{
    // }
    FragColor = vec4(result, 1.0);
    // if(FragPos.x > 0){
    //     FragColor.x = norm.x;
    //     FragColor.y = norm.y;
    //     FragColor.z = norm.z;
    // }
}