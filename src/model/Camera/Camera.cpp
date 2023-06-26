#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <vector>
#include <memory>

#include "../../controllers/Scene/Scene.hpp"

#include "../../maths/MMath.hpp"
#include "../../utils/Utils.hpp"
#include "../../utils/SceneSettings.hpp"
#include "../../utils/Projection.h"
#include "../../view/Lines.hpp"
#include "../../view/Gizmo.hpp"

#include "../../include/icons/IconsFontAwesome6.h"

#include "Camera.hpp"

using namespace glm;

Camera::Camera(Scene *scene, const std::string &name, const vec3 &position, const vec3 &target)
        : SceneObject{std::string("Camera"), SceneObjectTypes::CAMERA}, m_scene(scene), m_pos(position),
          m_target(target), m_imageTex(new Texture2D()), m_imagePlane(new Plane(scene)) {
    SetName(std::string(ICON_FA_CAMERA " ") + name);
    Initialize();
}

Camera::Camera(Scene *scene)
        : SceneObject{std::string("Camera"), SceneObjectTypes::CAMERA}, m_scene(scene), m_imageTex(new Texture2D()),
          m_imagePlane(new Plane(scene)) {
    SetName(std::string(ICON_FA_CAMERA " Camera"));
    /** Initialize camera's properties. */
    m_pos = vec3(4.0f, 4.0f, 4.0f);
    m_target = vec3(0.0f, 0.0f, 0.0f);

    Initialize();
}

void Camera::Initialize() {
    m_sceneSettings = m_scene->GetSceneSettings();

    /** Set the camera's resolution to the same as the viewport for now.*/
    m_resolution.x = m_sceneSettings->GetViewportWidth();
    m_resolution.y = m_sceneSettings->GetViewportHeight();

    /** Initialize camera's properties. */
    m_up = vec3(0.0f, 1.0f, 0.0f);
    m_forward = normalize(m_target - m_pos) * -1.0f;
    m_right = cross(m_forward, m_up);
    m_realUp = cross(m_forward, m_right) * -1.0f;
    m_projectionMatrix = perspective(
            radians(m_initialFoV),
            m_sceneSettings->GetViewportRatio(),
            m_near,
            m_far);


    mat4 K = mat4(1.0f);
    float fx = m_resolution.x / (m_sceneSettings->GetViewportRatio() * tan(radians(m_initialFoV) / 2.0f));
    float fy = m_resolution.y / (tan(radians(m_initialFoV) / 2.0f));
    K[0][0] = fx * 0.5f;
    K[1][1] = fy * 0.5f;
    K[2][0] = m_resolution.x / 2.0f;
    K[2][1] = m_resolution.y / 2.0f;
    K[2][2] =  -1.0f;
    K[3][3] =  -1.0f;

    m_volumeK = K;

    m_viewMatrix = lookAt(m_pos, m_target, m_up);

    /** Initialize cursor pos. */
    m_previousCursorPos = vec2(m_sceneSettings->GetViewportWidth() / 2, m_sceneSettings->GetViewportHeight() / 2);

    /** Parameters to visual components. */
    m_frustumLines = new Lines(m_scene, m_wireframeVertices, 16 * 3);
    if (m_frustumLines != nullptr) m_frustumLines->SetColor(1.0, 0.8, 0.8, 0.8);

    m_gizmo = new Gizmo(m_scene, m_pos, m_right, m_realUp, m_forward);

    /** Create the camera's image plane. */
    if (m_imagePlane != nullptr) m_imagePlane->SetTexture2D(m_imageTex);

    /** Center line. */
    m_centerLine = new Lines(m_scene, m_centerLineVertices, 6);
    if (m_centerLine != nullptr) m_centerLine->SetColor(0.0, 1.0, 1.0, 1.0);

    UpdateWireframe();
    UpdateGPUDescriptor();
}

Camera::~Camera() {
    delete m_frustumLines;
    delete m_centerLine;
    delete m_gizmo;
    delete m_imageTex;
    delete m_imagePlane;
}

const vec3 &Camera::GetPosition() {
    return m_pos;
}

void Camera::SetPosition(const vec3 &position) {
    m_pos = position;
    auto vm = lookAt(m_pos, m_target, m_up);
    SetExtrinsic(vm);
    UpdateWireframe();

    m_gizmo->SetPosition(position);
    m_gizmo->UpdateLines();

    /** Update center line. */
    WRITE_VEC3(m_centerLineVertices, 0, m_pos);
    m_centerLine->UpdateVertices(m_centerLineVertices);
}

void Camera::SetPosition(float x, float y, float z) {
    m_pos.x = x;
    m_pos.y = y;
    m_pos.z = z;
    auto vm = lookAt(m_pos, m_target, m_up);
    SetExtrinsic(vm);

    UpdateWireframe();

    m_gizmo->SetPosition(vec3(x, y, z));
    m_gizmo->UpdateLines();

    /** Update center line. */
    WRITE_VEC3(m_centerLineVertices, 0, m_pos);
    m_centerLine->UpdateVertices(m_centerLineVertices);

    UpdateGPUDescriptor();
}

const mat4 &Camera::GetViewMatrix() {
    return m_viewMatrix;
}

const mat4 &Camera::GetProjectionMatrix() {
    return m_projectionMatrix;
}

void Camera::ComputeMatricesFromInputs(GLFWwindow *window) {
//    static double lastTime = glfwGetTime();
    /** Compute time difference between current and last frame */
//    double currentTime = glfwGetTime();
    // Get mouse position
    double xpos = 0.0, ypos = 0.0;
    glfwGetCursorPos(window, &xpos, &ypos);

    // Reset mouse position for next frame
    if (m_sceneSettings->GetCameraModel() == CameraMovementModel::FPS) {}
    else {
        /** If the user is not left clicking, nothing happens in arc-ball mode. */
        if (!m_sceneSettings->GetMouseLeftClick()) {
            m_previousCursorPos.x = (float) xpos;
            m_previousCursorPos.y = (float) ypos;
//            lastTime = currentTime;
            return;
        }

        vec3 viewDir = -transpose(m_viewMatrix)[2];

        /** If the shift key is pressed, move is enabled and replaces the rotation. */
        float scrollSpeedCoef = max(1.5f / (0.5f + exp(0.2f * m_sceneSettings->GetScrollOffsets().y)), 0.2f);

        if (m_sceneSettings->GetAltKey()) {
            float deltaX = scrollSpeedCoef * 3.0f / (float) m_sceneSettings->GetViewportWidth();

            float xDisplacement = (float) (m_previousCursorPos.x - xpos) * deltaX;

            m_pos += xDisplacement * (m_pos - m_target);

            m_viewMatrix = lookAt(m_pos, m_target, m_up);

            m_previousCursorPos.x = (float) xpos;
            m_previousCursorPos.y = (float) ypos;
//            lastTime = currentTime;
            return;
        }

        if (m_sceneSettings->GetShiftKey()) {
            float deltaX = scrollSpeedCoef * 4.0f / (float) m_sceneSettings->GetViewportWidth();
            float deltaY = scrollSpeedCoef * 4.0f / (float) m_sceneSettings->GetViewportHeight();

            float xDisplacement = (float) (m_previousCursorPos.x - xpos) * deltaX;
            float yDisplacement = (float) (m_previousCursorPos.y - ypos) * deltaY;

            vec3 right = transpose(m_viewMatrix)[0];
            vec3 up = m_up;

            m_pos = m_pos + xDisplacement * right - up * yDisplacement;
            m_target = m_target + xDisplacement * right - up * yDisplacement;

            m_pos += scrollSpeedCoef * viewDir * (m_sceneSettings->GetScrollOffsets().y - m_prevScrollY);
            m_prevScrollY = m_sceneSettings->GetScrollOffsets().y;

            m_viewMatrix = lookAt(m_pos, m_target, m_up);

            m_previousCursorPos.x = (float) xpos;
            m_previousCursorPos.y = (float) ypos;
//            lastTime = currentTime;
            return;
        }

        // Get the homogenous position of the camera and pivot point
        vec4 position(m_pos.x, m_pos.y, m_pos.z, 1);
        vec4 pivot(m_target.x, m_target.y, m_target.z, 1);

        // step 1 : Calculate the amount of rotation given the mouse movement.
        auto deltaAngleX = (float) (2 * M_PI /
                                    m_sceneSettings->GetViewportWidth()); // a movement from left to right = 2*PI = 360 deg
        auto deltaAngleY = (float) (M_PI /
                                    m_sceneSettings->GetViewportHeight());    // a movement from top to bottom = PI = 180 deg
        float xAngle = (float) (m_previousCursorPos.x - xpos) * deltaAngleX;
        float yAngle = (float) (m_previousCursorPos.y - ypos) * deltaAngleY;

        // Extra step to handle the problem when the camera direction is the same as the up vector
        float cosAngle = dot(viewDir, m_up);
        if (cosAngle * Utils::Sign(yDeltaAngle) > 0.99f)
            yDeltaAngle = 0;

        // step 2: Rotate the camera around the pivot point on the first axis.
        mat4x4 rotationMatrixX(1.0f);
        rotationMatrixX = rotate(rotationMatrixX, xAngle, m_up);
        position = (rotationMatrixX * (position - pivot)) + pivot;

        // step 3: Rotate the camera around the pivot point on the second axis.
        mat4x4 rotationMatrixY(1.0f);
        vec3 right = transpose(m_viewMatrix)[0];
        rotationMatrixY = rotate(rotationMatrixY, yAngle, right);
        vec3 finalPosition = (rotationMatrixY * (position - pivot)) + pivot;

        // Update the camera view (we keep the same look-at and the same up vector)
        m_pos = finalPosition + scrollSpeedCoef * viewDir * (m_sceneSettings->GetScrollOffsets().y - m_prevScrollY);
        m_prevScrollY = m_sceneSettings->GetScrollOffsets().y;

        m_viewMatrix = lookAt(
                m_pos,
                m_target,
                m_up);
    }

    // Update the mouse position for the next rotation
    m_previousCursorPos.x = (float) xpos;
    m_previousCursorPos.y = (float) ypos;

}

const vec3 &Camera::GetTarget() { return m_target; }

void Camera::SetTarget(const vec3 &target) {
    m_target = target;
    m_viewMatrix = lookAt(m_pos, m_target, m_up);
    UpdateWireframe();
    m_forward = normalize(m_pos - m_target) * -1.0f;
    m_right = normalize(cross(m_forward, m_up));
    m_realUp = normalize(cross(m_forward, m_right)) * -1.0f;

    m_gizmo->SetX(m_right * 0.5f);
    m_gizmo->SetY(m_realUp * 0.5f);
    m_gizmo->SetZ(m_forward * 0.5f);
    m_gizmo->UpdateLines();

    /** Update center line. */
    WRITE_VEC3(m_centerLineVertices, 3, m_pos + m_forward * m_centerLineLength);
    m_centerLine->UpdateVertices(m_centerLineVertices);
}

const vec3 &Camera::GetRight() { return m_right; }

void Camera::SetRight(const vec3 &v) {
    m_right = v;
    UpdateWireframe();
    m_gizmo->SetX(normalize(m_right) * 0.5f);
    m_gizmo->UpdateLines();
}

const vec3 &Camera::GetUp() { return m_up; }

void Camera::SetUp(const vec3 &v) {
    m_up = v;
    UpdateWireframe();
    m_gizmo->SetY(normalize(v));
    m_gizmo->UpdateLines();
}

const float *Camera::GetWireframe() {
    return m_wireframeVertices;
}

void Camera::UpdateWireframe() {
    vec3 corner_top_left_tmp = NDCToCamera(vec2(-1.0, 1.0), m_volumeK) * m_wireSize;
//    corner_top_left_tmp.z *= -1.0f;
    vec3 corner_top_left = CameraToWorld(vec4(corner_top_left_tmp, 1.0f), m_viewMatrix);

    vec3 corner_top_right_tmp = NDCToCamera(vec2(1.0, 1.0), m_volumeK) * m_wireSize;
//    corner_top_right_tmp.z *= -1.0f;
    vec3 corner_top_right = CameraToWorld(vec4(corner_top_right_tmp, 1.0f), m_viewMatrix);

    vec3 corner_bot_left_tmp = NDCToCamera(vec2(-1.0, -1.0), m_volumeK) * m_wireSize;
//    corner_bot_left_tmp.z *= -1.0f;
    vec3 corner_bot_left = CameraToWorld(vec4(corner_bot_left_tmp, 1.0f), m_viewMatrix);

    vec3 corner_bot_right_tmp = NDCToCamera(vec2(1.0, -1.0), m_volumeK) * m_wireSize;
//    corner_bot_right_tmp.z *= -1.0f;
    vec3 corner_bot_right = CameraToWorld(vec4(corner_bot_right_tmp, 1.0f), m_viewMatrix);

    if (m_imagePlane != nullptr)
        m_imagePlane->SetVertices(corner_bot_left, corner_bot_right, corner_top_left, corner_top_right);
//    if(m_imagePlane != nullptr) m_imagePlane->SetVertices(corner_top_left, corner_top_right, corner_bot_left, corner_bot_right);

    WRITE_VEC3(m_wireframeVertices, 0, corner_top_left);
    WRITE_VEC3(m_wireframeVertices, 3, corner_top_right);

    WRITE_VEC3(m_wireframeVertices, 6, corner_bot_left);
    WRITE_VEC3(m_wireframeVertices, 9, corner_bot_right);

    WRITE_VEC3(m_wireframeVertices, 12, corner_top_left);
    WRITE_VEC3(m_wireframeVertices, 15, corner_bot_left);

    WRITE_VEC3(m_wireframeVertices, 18, corner_top_right);
    WRITE_VEC3(m_wireframeVertices, 21, corner_bot_right);

    WRITE_VEC3(m_wireframeVertices, 24, m_pos);
    WRITE_VEC3(m_wireframeVertices, 27, corner_top_left);

    WRITE_VEC3(m_wireframeVertices, 30, m_pos);
    WRITE_VEC3(m_wireframeVertices, 33, corner_top_right);

    WRITE_VEC3(m_wireframeVertices, 36, m_pos);
    WRITE_VEC3(m_wireframeVertices, 39, corner_bot_left);

    WRITE_VEC3(m_wireframeVertices, 42, m_pos);
    WRITE_VEC3(m_wireframeVertices, 45, corner_bot_right);

    m_frustumLines->UpdateVertices(m_wireframeVertices);
}

float Camera::GetFovX() const { return m_fov.x; }

void Camera::SetFovX(float fov, bool keepRatio) {
    m_fov.x = fov;
    UpdateWireframe();
    // TODO: ratio.
}

float Camera::GetFovY() const { return m_fov.y; }

void Camera::SetFovY(float fov) {
    m_fov.y = fov;
    UpdateWireframe();
}

float Camera::GetNear() const { return m_near; }

void Camera::SetNear(float near) {
    m_near = near;
    UpdateWireframe();
}

void Camera::SetFar(float far) {
    m_far = far;
    UpdateWireframe();
}

float Camera::GetFar() const { return m_far; }

void Camera::Render() {
    glDisable(GL_BLEND);
    /** Frustum */
    if (m_showFrustumLines && m_frustumLines != nullptr) m_frustumLines->Render();
    if(m_gizmo!= nullptr) m_gizmo->Render();

    /** Image plane linked. */
    if (m_imagePlane != nullptr) {
        if(m_cudaTexture != nullptr){
            m_imagePlane->SetUseCustomTex(true);
            m_imagePlane->SetCustomTex(m_cudaTexture->GetTex());
            m_imagePlane->Render();
        }else{
            m_imagePlane->SetUseCustomTex(false);
            m_imagePlane->Render();
        }
    }

    /** Center line. */
    if (m_displayCenterLine) {
        m_centerLine->Render();
    }

    /** Rays (partial) for each pixel. */
    glEnable(GL_BLEND);
}

void Camera::InitializeCudaTexture(){
    m_cudaTexture = std::make_shared<CudaTexture>(m_resolution.x, m_resolution.y);
    m_imagePlane->SetCustomTex(m_cudaTexture->GetTex());
    m_imagePlane->SetUseCustomTex(true);
}

std::shared_ptr<CudaTexture> Camera::GetCudaTexture(){
    return m_cudaTexture;
}

void Camera::SetIntrinsic(const mat4 &intrinsic) {
    m_projectionMatrix = intrinsic;
    m_volumeK = intrinsic;

    UpdateWireframe();
    UpdateGPUDescriptor();
}

void Camera::SetExtrinsic(const mat4 &extrinsic) {
    m_viewMatrix = extrinsic;

    mat3 rotMat(m_viewMatrix);
    vec3 d(m_viewMatrix[3]);

    m_forward = transpose(m_viewMatrix)[2] * -1.0f;
    m_right = transpose(m_viewMatrix)[0];
    m_up = vec3(0.0, 1.0, 0.0);
    m_realUp = normalize(cross(m_forward, m_right)) * -1.0f;

    m_pos = -d * rotMat;

    m_gizmo->SetPosition(m_pos);
    m_gizmo->SetX(m_right * 0.5f);
    m_gizmo->SetY(m_realUp * 0.5f);
    m_gizmo->SetZ(m_forward * 0.5f);
    m_gizmo->UpdateLines();

    UpdateWireframe();

    /** Update center line. */
    WRITE_VEC3(m_centerLineVertices, 0, m_pos);
    WRITE_VEC3(m_centerLineVertices, 3, m_pos + m_forward * m_centerLineLength);
    m_centerLine->UpdateVertices(m_centerLineVertices);

    UpdateGPUDescriptor();
}

const mat4 &Camera::GetIntrinsic() {
//    auto aspect = (float)m_resolution.x / (float)m_resolution.y;
//    mat4 K = mat4(1.0f);
//    float fx = m_resolution.x / (2.0 * tan(m_initialFoV / 2.0));
//    float fy = m_resolution.y / (2.0 * tan(m_initialFoV  / 2.0));
//    K[0][0] = fx;
//    K[1][1] = fy * aspect;
//    K[2][0] = m_resolution.x / 2.0;
//    K[2][1] = m_resolution.y / 2.0;
//    K[2][2] = -1.0f;
//
//    m_volumeK = K;
    return m_volumeK;
}

const mat4 &Camera::GetExtrinsic() {
    return m_viewMatrix;
}

const ivec2 &Camera::GetResolution() {
    return m_resolution;
}

void Camera::SetImage(Image *image) {
    if (image == nullptr) return;
    m_imageTex->LoadFromImage(image);
    filename = image->filename;
    m_showImagePlane = true;
}

bool Camera::IsCenterLineVisible() const {
    return m_displayCenterLine;
}

void Camera::SetIsCenterLineVisible(bool visible) {
    m_displayCenterLine = visible;
}

void Camera::SetCenterLineLength(float length) {
    m_centerLineLength = length;

    /** Update center line. */
    WRITE_VEC3(m_centerLineVertices, 3, m_pos + m_forward * m_centerLineLength);
    m_centerLine->UpdateVertices(m_centerLineVertices);
}

float Camera::GetCenterLineLength() const {
    return m_centerLineLength;
}

void Camera::SetShowFrustumLines(bool value) {
    m_showFrustumLines = value;
}

bool Camera::ShowFrustumLines() const {
    return m_showFrustumLines;
}

bool Camera::ShowImagePlane() const {
    return m_showImagePlane;
}

void Camera::SetShowImagePlane(bool visible) {
    m_showImagePlane = visible;
}

void Camera::SetFrustumSize(float value) {
    m_wireSize = value;
    UpdateWireframe();
}

void Camera::UpdateGPUDescriptor() {
    m_desc.Host()->camPos = m_pos;
    m_desc.Host()->camExt = m_viewMatrix;
    m_desc.Host()->camInt = m_volumeK;
    m_desc.Host()->width = m_resolution.x;
    m_desc.Host()->height = m_resolution.y;
    m_desc.ToDevice();
}

void Camera::SetResolution(const ivec2& res) {
    m_resolution = res;
    UpdateGPUDescriptor();
}

CameraDescriptor *Camera::GetGPUDescriptor() {
    return m_desc.Device();
}

GPUData<IntegrationRangeDescriptor>& Camera::GetIntegrationRangeGPUDescriptor(){
    return m_integrationDesc;
}

GPUData<CameraDescriptor>& Camera::GetGPUData(){
    return m_desc;
}