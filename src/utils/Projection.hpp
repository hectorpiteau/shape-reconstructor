#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace Projection
{   
    /**
     * @brief Project camera (3D) coordinates in World coordinates (3D).
     * 
     * @param vec 
     * @param extrinsics 
     * @return glm::vec4 
     */
    glm::vec4 CameraToWorld(glm::vec4 vec, glm::mat4 extrinsics)
    {
        /** Pw = R^t @ Pc + T */
        /** PointWorld = Rotation^Transposed multiplied by PointCamera + (Translation from world to camera origins)  */
        glm::mat4 ext = glm::transpose(extrinsics);
        ext[0][3] = 0;
        ext[1][3] = 0;
        ext[2][3] = 0;
        ext[3][0] = 0;
        ext[3][1] = 0;
        ext[3][2] = 0;
        ext[3][3] = 1;

        glm::mat4 trans = glm::mat4(0.0);
        trans[0][0] = 1.0;
        trans[1][1] = 1.0;
        trans[2][2] = 1.0;
        trans[3][3] = 1.0;
        trans[0][3] = -extrinsics[0][3];
        trans[1][3] = -extrinsics[1][3];
        trans[2][3] = -extrinsics[2][3];

        return ext * trans * vec;
    }

    /**
     * @brief Project World coordinates to Camera coordinates.
     * 
     * @param worldCoords 
     * @param extrinsics 
     * @return glm::vec4 
     */
    glm::vec4 WorldToCamera(glm::vec4 worldCoords, glm::mat4 extrinsics){
        return extrinsics * worldCoords;
    }

    /**
     * @brief Project Camera coordinates (3D) to image coordinates (2D image plane).
     * 
     * @param cameraCoords 
     * @param intrinsicsImage 
     * @return glm::vec3 
     */
    glm::vec3 CameraToImage(glm::vec4 cameraCoords, glm::mat3x4 intrinsicsImage)
    {
        return intrinsicsImage * cameraCoords;
    }

    /**
     * @brief Project Image coordinates to Pixel coordinates.
     * 
     * @param imageCoords 
     * @param intrinsicsPixel 
     * @return glm::vec2 
     */
    glm::vec2 ImageToPixel(glm::vec3 imageCoords, glm::mat3 intrinsicsPixel)
    {
        return intrinsicsPixel * imageCoords;
    }

    /**
     * @brief Project Camera coordinates to Pixel's coordinates.
     * 
     * @param cameraCoords 
     * @param intrinsics 
     * @return glm::vec2 
     */
    glm::vec2 CameraToPixel(glm::vec3 cameraCoords, glm::mat3 intrinsics)
    {
        return intrinsics * cameraCoords; //TODO: int round? 
    }

    /**
     * @brief Create an intrinsics matrix from parameters without skew.
     * 
     * @param focalLengthX 
     * @param focalLengthY 
     * @param pixelSize
     * @param pixelTranslation 
     * @return glm::mat3 
     */
    glm::mat3 BuildIntrinsics(double focalLengthX, double focalLengthY, glm::vec2 pixelSize, glm::vec2 pixelTranslation){
        glm::mat3 res(0.0);
        res[0][0] = focalLengthX/pixelSize.x;
        res[1][1] = focalLengthY/pixelSize.y;
        res[0][2] = pixelTranslation.x;
        res[1][2] = pixelTranslation.y;
        res[3][3] = 1.0;
        return res;
    }

    /**
     * @brief Get the pixels sizes. It depends on the sensor's size which may be known. 
     * 
     * @param width : The amount of pixels in the width of the image.
     * @param height : The amount of pixels in the height of the image.
     * @param sensorWidth : The sensor's width in mm (scene unit). 
     * @param sensorHeight : The sensor's height in mm (scene unit).
     * 
     * @return glm::vec2 The vector containing the size of one pixel in the scene's unit.
     */
    glm::vec2 GetPixelSizeVec(int width, int height, double sensorWidth, double sensorHeight){
        return glm::vec2(sensorWidth/width, sensorHeight/height);
    }

};