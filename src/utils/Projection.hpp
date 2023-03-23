#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace Projection
{   
    /**
     * @brief Project camera (3D) coordinates in World coordinates (3D).
     * 
     * @param vec : The point expressed in the camera coordinate system.
     * @param extrinsics : The extrinsics matrix of the camera.
     * @param cameraPosition : The position of the camera in world space.
     * @return glm::vec4 : The same point as vec but expressed with respect 
     * to the world space coordinates system.
     */
    glm::vec4 CameraToWorld(const glm::vec4 &vec, const glm::mat4 &extrinsics, const glm::vec3 &cameraPosition)
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

        return ext * trans * vec + glm::vec4(cameraPosition, 1.0f);
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
    
    /**
     * @brief Converts coordinates expressed in (u,v) pixel space which is in 
     * range u \in [0, height], v \in [0, width], into coordinates in Normalized 
     * Device Coordinates space : [-1, 1] for both coordinates.
     * 
     * @param pixelCoords : The pixel coordinates in image space, the origin is top-left corner. 
     * @param width : The amount of pixels in the width.
     * @param height : The amount of pixels in the height.
     * @return glm::vec2 : A vector in Normalized Device Coordinates ([-1, 1], [-1, 1])
     */
    glm::vec2 PixelToNDC(const glm::vec2 &pixelCoords, const glm::mat4 &intrinsics, int width, int height){
        glm::vec2 tmp = (2.0f * pixelCoords / glm::vec2(intrinsics[0][0], intrinsics[1][1])) - glm::vec2(1.0f);
        // Account for image aspect ratio
        tmp.x *= (intrinsics[1][1] / intrinsics[0][0]); //TODO: division can be moved sooner in code.
        return tmp;
    }

    /**
     * @brief Converts coordinates expressed in (u,v) pixel space which is in 
     * range u \in [0, height], v \in [0, width], into coordinates in Normalized 
     * Device Coordinates space : [-1, 1] for both coordinates.
     * 
     * @param pixelCoords : The pixel coordinates in image space, the origin is top-left corner. 
     * @param distortion : 
     * @param width : The amount of pixels in the width.
     * @param height : The amount of pixels in the height.
     * @return glm::vec2 : A vector in Normalized Device Coordinates ([-1, 1], [-1, 1])
     */
    glm::vec2 PixelToNDC(const glm::vec2 &pixelCoords, const glm::mat4 &intrinsics, const glm::vec2 &distortion, int width, int height){
        glm::vec2 tmp = (2.0f * pixelCoords / glm::vec2(intrinsics[0][0], intrinsics[1][1])) - glm::vec2(1.0f);
        // Account for image aspect ratio
        tmp.x *= (intrinsics[1][1] / intrinsics[0][0]); //TODO: division can be moved sooner in code.
        return tmp;
    }

    /**
     * @brief 
     * 
     * @param ndcCoords 
     * @param intrinsics 
     * @param width 
     * @param height 
     * @return glm::vec2 
     */
    glm::vec2 NDCToPixel(const glm::vec2 &ndcCoords, const glm::mat4 &intrinsics, int width, int height){
        // Calculate pixel coordinates
        return glm::vec2(
            (ndcCoords.x + 1.0f) * (intrinsics[0][0] / 2.0f),
            (ndcCoords.y + 1.0f) * (intrinsics[1][1] / 2.0f)
        );
    }

};