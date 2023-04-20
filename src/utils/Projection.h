
#include "../cuda/Projection.cuh"

#ifndef PROJECTION_CUDA_H

#ifndef PROJECTION_H
#define PROJECTION_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace glm;

/**
 * @brief Project camera (3D) coordinates in World coordinates (3D).
 *
 * @param vec : The point expressed in the camera coordinate system.
 * @param extrinsic : The extrinsic matrix of the camera.
 * @param cameraPosition : The position of the camera in world space.
 * @return vec4 : The same point as vec but expressed with respect
 * to the world space coordinates system.
 */
inline vec4 CameraToWorld(const vec4 &vec, const mat4 &extrinsic)
{
    /** Pw = R^t @ Pc + T */
    /** PointWorld = Rotation^Transposed multiplied by PointCamera + (Translation from world to camera origins)  */
    mat4 ext = transpose(extrinsic);
    ext[0][3] = 0;
    ext[1][3] = 0;
    ext[2][3] = 0;
    ext[3][0] = 0;
    ext[3][1] = 0;
    ext[3][2] = 0;
    ext[3][3] = 1;

    mat4 trans = mat4(0.0);
    trans[0][0] = 1.0;
    trans[1][1] = 1.0;
    trans[2][2] = 1.0;
    trans[3][3] = 1.0;
    trans[3][0] = -extrinsic[3][0];
    trans[3][1] = -extrinsic[3][1];
    trans[3][2] = -extrinsic[3][2];

    return ext * trans * vec;
}

/**
 * @brief Project World coordinates to Camera coordinates.
 *
 * @param worldCoords
 * @param extrinsic
 * @return vec4
 */
vec4 WorldToCamera(vec4 worldCoords, mat4 extrinsic)
{
    return extrinsic * worldCoords;
}

/**
 * @brief Project Camera coordinates (3D) to image coordinates (2D image plane).
 *
 * @param cameraCoords
 * @param intrinsicImage
 * @return vec3
 */
vec3 CameraToImage(vec4 cameraCoords, mat3x4 intrinsicImage)
{
    return intrinsicImage * cameraCoords;
}

/**
 * @brief Project Image coordinates to Pixel coordinates.
 *
 * @param imageCoords
 * @param intrinsicPixel
 * @return vec2
 */
vec2 ImageToPixel(vec3 imageCoords, mat3 intrinsicPixel)
{
    return intrinsicPixel * imageCoords;
}

/**
 * @brief Project Camera coordinates to Pixel's coordinates.
 *
 * @param cameraCoords
 * @param intrinsic
 * @return vec2
 */
vec2 CameraToPixel(vec3 cameraCoords, mat3 intrinsic)
{
    return intrinsic * cameraCoords; // TODO: int round?
}

/**
 * @brief Convert normalized Device Coordinates to Camera coordinates.
 *
 * @param ndcCoords : Normalized Device Coordinates in range [-1, 1] for eacch elements.
 * @param intrinsic : The camera's intrinsic matrix.
 * @return vec3 : A vec3 that correspond to the point in ndcCoords but in the
 * camera coordinate system.
 */
vec3 NDCToCamera(const vec2 &ndcCoords, const mat4 &intrinsic)
{
    return vec3(ndcCoords.x * 1 / intrinsic[0][0], ndcCoords.y * 1 / intrinsic[1][1], 1.0);
}

/**
 * @brief Create an intrinsic matrix from parameters without skew.
 *
 * @param focalLengthX
 * @param focalLengthY
 * @param pixelSize
 * @param pixelTranslation
 * @return mat3
 */
mat3 Buildintrinsic(double focalLengthX, double focalLengthY, vec2 pixelSize, vec2 pixelTranslation)
{
    mat3 res(0.0);
    res[0][0] = focalLengthX / pixelSize.x;
    res[1][1] = focalLengthY / pixelSize.y;
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
 * @return vec2 The vector containing the size of one pixel in the scene's unit.
 */
vec2 GetPixelSizeVec(int width, int height, double sensorWidth, double sensorHeight)
{
    return vec2(sensorWidth / width, sensorHeight / height);
}

/**
 * @brief Converts coordinates expressed in (u,v) pixel space which is in
 * range u \in [0, height], v \in [0, width], into coordinates in Normalized
 * Device Coordinates space : [-1, 1] for both coordinates.
 *
 * @param pixelCoords : The pixel coordinates in image space, the origin is top-left corner.
 * @param width : The amount of pixels in the width.
 * @param height : The amount of pixels in the height.
 * @return vec2 : A vector in Normalized Device Coordinates ([-1, 1], [-1, 1])
 */
inline vec2 PixelToNDC(const vec2 &pixelCoords, const mat4 &intrinsic, int width, int height)
{
    vec2 tmp = (2.0f * pixelCoords / vec2(intrinsic[0][0], intrinsic[1][1])) - vec2(1.0f);
    // Account for image aspect ratio
    tmp.x *= (intrinsic[1][1] / intrinsic[0][0]); // TODO: division can be moved sooner in code.
    return tmp;
}

inline vec3 PixelToWorld(const vec2 &pixelCoords, const mat4 &intrinsic, const mat4 &extrinsic, int width, int height)
{
    auto ndc = PixelToNDC(pixelCoords,
                          intrinsic,
                          width,
                          height);

    auto cam = vec4(NDCToCamera(ndc, intrinsic), 1.0f);

    return CameraToWorld(cam, extrinsic);
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
 * @return vec2 : A vector in Normalized Device Coordinates ([-1, 1], [-1, 1])
 */
inline vec2 PixelToNDC(const vec2 &pixelCoords, const mat4 &intrinsic, const vec2 &distortion, int width, int height)
{
    vec2 tmp = (2.0f * pixelCoords / vec2(intrinsic[0][0], intrinsic[1][1])) - vec2(1.0f);
    // Account for image aspect ratio
    tmp.x *= (intrinsic[1][1] / intrinsic[0][0]); // TODO: division can be moved sooner in code.
    return tmp;
}

/**
 * @brief
 *
 * @param ndcCoords
 * @param intrinsic
 * @param width
 * @param height
 * @return vec2
 */
inline vec2 NDCToPixel(const vec2 &ndcCoords, const mat4 &intrinsic, int width, int height)
{
    // Calculate pixel coordinates
    return vec2(
        (ndcCoords.x + 1.0f) * (intrinsic[0][0] / 2.0f),
        (ndcCoords.y + 1.0f) * (intrinsic[1][1] / 2.0f));
}

/**
 * @brief
 *
 * @param worldCoords : A point in the world coordinates
 * @param srcExtrinsic
 * @param dstExtrinsic
 * @param srcIntrinsic
 * @param dstIntrinsic
 * @return vec2
 */
inline vec2 CamToCamHomography(
    const vec3 &srcCoords,
    float distanceFromSrc,
    const mat4 &srcExtrinsic,
    const mat4 &dstExtrinsic,
    const mat4 &srcIntrinsic,
    const mat4 &dstIntrinsic)
{
    return vec2(0.0, 0.0);
}

#endif //PROJECTION_H

#endif //PROJECTION_CUDA_H