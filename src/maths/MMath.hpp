#ifndef _MMATH_H_
#define _MMATH_H_
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

class MMath {
public:

MMath(){};

static float DegreeToRadian(float degree){
    return degree * (M_PI / 180);
}

static double DegreeToRadian(double degree){
    return degree * (M_PI / 180);
}

static void InitializeMat4ForScale(glm::mat4 &mat, double x, double y, double z){
    mat = glm::mat4(0.0);
    mat[0][0] = x;   
    mat[1][1] = y;   
    mat[2][2] = z;   
    mat[3][3] = 1.0;   
}

static void InitializeMat4ForRotation(glm::mat4 &mat, double x, double y, double z){
    mat = glm::mat4(0.0);
    glm::mat4 rx(0.0), ry(0.0), rz(0.0);

    float rad_x = MMath::DegreeToRadian(x);
    float rad_y = MMath::DegreeToRadian(y);
    float rad_z = MMath::DegreeToRadian(z);

    rx[0][0] = 1.0;
    rx[1][1] = cosf(rad_x);
    rx[2][1] = sinf(rad_x);
    rx[1][2] = -sinf(rad_x);
    rx[2][2] = cosf(rad_x);
    rx[3][3] = 1.0;

    ry[0][0] = cosf(rad_y);
    ry[2][0] = sinf(rad_y);
    ry[1][1] = 1.0;
    ry[0][2] = -sinf(rad_y);
    ry[2][2] = cosf(rad_y);
    ry[3][3] = 1.0;

    rz[0][0] = cosf(rad_z);
    rz[1][0] = sinf(rad_z);
    rz[0][1] = -sinf(rad_z);
    rz[1][1] = cosf(rad_z);
    rz[2][2] = 1.0;
    rz[3][3] = 1.0;

    mat = rz * ry * rx;
}

static void InitializeMat4ForTranslate(glm::mat4 &mat, double x, double y, double z){
    mat = glm::mat4(0.0);
    
    mat[0][0] = 1.0;   
    mat[1][1] = 1.0;   
    mat[2][2] = 1.0;   
    mat[3][3] = 1.0;   
    
    mat[3][0] = x;   
    mat[3][1] = y;   
    mat[3][2] = z;   
}

// static glm::vec3 Cross(const glm::vec3& mat, const glm::vec3& v)
// {
//     const float _x = mat.y * v.z - mat.z * v.y;
//     const float _y = mat.z * v.x - mat.x * v.z;
//     const float _z = mat.x * v.y - mat.y * v.x;

//     return glm::vec3(_x, _y, _z);
// }

static glm::mat4 InitCameraTransform(const glm::vec3& Target, const glm::vec3& Up)
{
    glm::vec3 N = glm::normalize(Target);
    glm::vec3 U = glm::normalize(glm::cross(Up, N));
    glm::vec3 V = glm::cross(N, U);

    glm::mat4 res(0.0);

    res[0][0] = U.x;   res[0][1] = U.y;   res[0][2] = U.z;
    res[1][0] = V.x;   res[1][1] = V.y;   res[1][2] = V.z;
    res[2][0] = N.x;   res[2][1] = N.y;    res[2][2] = N.z;
    res[3][3] = 1.0f;

    return res;
}


static glm::mat4 InitCameraTransform(const glm::vec3& Pos, const glm::vec3& Target, const glm::vec3& Up)
{
    glm::vec3 N = glm::normalize(Target);
    glm::vec3 U = glm::normalize(glm::cross(Up, N));
    glm::vec3 V = glm::cross(N, U);

    glm::mat4 res(0.0);

    res[0][0] = U.x;   res[0][1] = U.y;   res[0][2] = U.z; res[0][3] = -Pos.x;
    res[1][0] = V.x;   res[1][1] = V.y;   res[1][2] = V.z; res[1][3] = -Pos.y;
    res[2][0] = N.x;   res[2][1] = N.y;   res[2][2] = N.z; res[2][3] = -Pos.z;
    res[3][3] = 1.0f;

    return res;
}

};



#endif // _MMATH_H_