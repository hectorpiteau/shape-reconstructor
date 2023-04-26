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

Camera::Camera(Scene *scene, const std::string& name, const vec3& position, const vec3& target)
    : SceneObject{std::string("Camera"), SceneObjectTypes::CAMERA}, m_scene(scene), m_pos(position), m_target(target)
{
    SetName(std::string(ICON_FA_CAMERA " ") + name);
    Initialize();
}

Camera::Camera(Scene *scene)
    : SceneObject{std::string("Camera"), SceneObjectTypes::CAMERA}, m_scene(scene)
{
    SetName(std::string(ICON_FA_CAMERA " Camera"));
    /** Initialize camera's properties. */
    m_pos = vec3(4.0f, 4.0f, 4.0f);
    m_target = vec3(0.0f, 0.0f, 0.0f);
    Initialize();
}

void Camera::Initialize(){
    m_sceneSettings = m_scene->GetSceneSettings();
    /** Initialize camera's properties. */
    m_up = vec3(0.0f, 1.0f, 0.0f);
    m_forward = normalize(m_target - m_pos);
    m_right = cross(m_forward, m_up);
    m_realUp = cross(m_forward, m_right);
    m_projectionMatrix = perspective(
        radians(m_initialFoV),
        m_sceneSettings->GetViewportRatio(),
        m_near,
        m_far);
    m_viewMatrix = lookAt(m_pos, m_target, m_up);

    /** Initialize cursor pos. */
    m_previousCursorPos = vec2(m_sceneSettings->GetViewportWidth() / 2, m_sceneSettings->GetViewportHeight() / 2);

    /** Parameters to visual components. */
    m_frustumLines = new Lines(m_scene, m_wireframeVertices, 16 * 3);
    m_frustumLines->SetColor(1.0, 0.8, 0.8, 0.8);
    m_gizmo = new Gizmo(m_scene, m_pos, m_right, m_realUp, m_forward);

    /** Create the camera's image plane. */
    m_imageTex = new Texture2D();
    m_imagePlane = new Plane(m_scene);
    m_imagePlane->SetTexture2D(m_imageTex);

    /** Center line. */
    m_centerLine = new Lines(m_scene, m_centerLineVertices, 6);
    m_centerLine->SetColor(0.0, 1.0, 1.0, 1.0);
}

Camera::~Camera()
{
    delete m_frustumLines;
    delete m_centerLine;
    delete m_gizmo;
    delete m_imageTex;
    delete m_imagePlane;
}

const vec3 &Camera::GetPosition()
{
    return m_pos;
}

void Camera::SetPosition(const vec3 &position)
{
    m_pos = position;
    m_viewMatrix = lookAt(m_pos, m_target, m_up);
    UpdateWireframe();
    
    m_gizmo->SetPosition(position);
    m_gizmo->UpdateLines();

    /** Update center line. */
    WRITE_VEC3(m_centerLineVertices, 0, m_pos);
    m_centerLine->UpdateVertices(m_centerLineVertices);
}

void Camera::SetPosition(float x, float y, float z)
{
    m_pos.x = x;
    m_pos.y = y;
    m_pos.z = z;
    m_viewMatrix = lookAt(m_pos, m_target, m_up);
    UpdateWireframe();
    m_gizmo->SetPosition(vec3(x,y,z));
    m_gizmo->UpdateLines();

    /** Update center line. */
    WRITE_VEC3(m_centerLineVertices, 0, m_pos);
    m_centerLine->UpdateVertices(m_centerLineVertices);
}

const mat4 &Camera::GetViewMatrix()
{
    return m_viewMatrix;
}

const mat4 &Camera::GetProjectionMatrix()
{
    return m_projectionMatrix;
}

void Camera::ComputeMatricesFromInputs(GLFWwindow *window)
{

    static double lastTime = glfwGetTime();

    /** Compute time difference between current and last frame */
    double currentTime = glfwGetTime();

    float deltaTime = float(currentTime - lastTime);

    // Get mouse position
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    // Reset mouse position for next frame
    if (m_sceneSettings->GetCameraModel() == CameraMovementModel::FPS)
    {
        // glfwSetCursorPos(window, m_sceneSettings->GetViewportWidth() / 2, m_sceneSettings->GetViewportHeight() / 2);
        // // Compute new orientation
        // m_horizontalAngle += m_mouseSpeed * float(m_sceneSettings->GetViewportWidth() / 2 - xpos);
        // m_verticalAngle += m_mouseSpeed * float(m_sceneSettings->GetViewportHeight() / 2 - ypos);
        // // Direction : Spherical coordinates to Cartesian coordinates conversion
        // vec3 direction(
        //     cos(m_verticalAngle) * sin(m_horizontalAngle),
        //     sin(m_verticalAngle),
        //     cos(m_verticalAngle) * cos(m_horizontalAngle));

        // // Right vector
        // vec3 right = vec3(
        //     sin(m_horizontalAngle - 3.14f / 2.0f),
        //     0,
        //     cos(m_horizontalAngle - 3.14f / 2.0f));

        // // Up vector
        // vec3 up = cross(right, direction);

        // // Move forward
        // if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        // {
        //     m_pos += direction * deltaTime * m_speed;
        // }
        // // Move backward
        // if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        // {
        //     m_pos -= direction * deltaTime * m_speed;
        // }
        // // Strafe right
        // if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        // {
        //     m_pos += right * deltaTime * m_speed;
        // }
        // // Strafe left
        // if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        // {
        //     m_pos -= right * deltaTime * m_speed;
        // }

        // // Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
        // // (intrinsics)
        // // m_projectionMatrix = perspective(radians(m_initialFoV), m_sceneSettings->GetViewportRatio(), 0.1f, 100.0f);

        // // Camera matrix
        // // (extrinsics)
        // m_viewMatrix = lookAt(
        //     m_pos,             // Camera is here
        //     m_pos + direction, // and looks here : at the same position, plus "direction"
        //     m_up               // Head is up (set to 0,-1,0 to look upside-down)
        // );
    }
    else
    {
        /** If the user is not left clicking, nothing happens in arcball mode. */
        if (m_sceneSettings->GetMouseLeftClick() == false)
        {
            m_previousCursorPos.x = xpos;
            m_previousCursorPos.y = ypos;
            lastTime = currentTime;
            return;
        }

        vec3 viewDir = -transpose(m_viewMatrix)[2];

        /** If the shift key is pressed, move is enabled and replaces the rotation. */
        float scrollSpeedCoef = max(1.5f / (0.5f + exp(0.2f * m_sceneSettings->GetScrollOffsets().y)), 0.2f);

        if (m_sceneSettings->GetShiftKey())
        {

            float deltaX = scrollSpeedCoef * 4.0f / m_sceneSettings->GetViewportWidth();
            float deltaY = scrollSpeedCoef * 4.0f / m_sceneSettings->GetViewportHeight();

            float xDisplacement = (m_previousCursorPos.x - xpos) * deltaX;
            float yDisplacement = (m_previousCursorPos.y - ypos) * deltaY;

            vec3 right = transpose(m_viewMatrix)[0];
            vec3 up = m_up;

            m_pos = m_pos + xDisplacement * right - up * yDisplacement;
            m_target = m_target + xDisplacement * right - up * yDisplacement;

            m_pos += scrollSpeedCoef * viewDir * (m_sceneSettings->GetScrollOffsets().y - m_prevScrollY);
            m_prevScrollY = m_sceneSettings->GetScrollOffsets().y;

            m_viewMatrix = lookAt(m_pos, m_target, m_up);

            m_previousCursorPos.x = xpos;
            m_previousCursorPos.y = ypos;

            lastTime = currentTime;
            return;
        }

        // Get the homogenous position of the camera and pivot point
        vec4 position(m_pos.x, m_pos.y, m_pos.z, 1);
        vec4 pivot(m_target.x, m_target.y, m_target.z, 1);

        // step 1 : Calculate the amount of rotation given the mouse movement.
        float deltaAngleX = (2 * M_PI / m_sceneSettings->GetViewportWidth()); // a movement from left to right = 2*PI = 360 deg
        float deltaAngleY = (M_PI / m_sceneSettings->GetViewportHeight());    // a movement from top to bottom = PI = 180 deg
        float xAngle = (m_previousCursorPos.x - xpos) * deltaAngleX;
        float yAngle = (m_previousCursorPos.y - ypos) * deltaAngleY;

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

        // Update the camera view (we keep the same lookat and the same up vector)
        m_pos = finalPosition + scrollSpeedCoef * viewDir * (m_sceneSettings->GetScrollOffsets().y - m_prevScrollY);
        m_prevScrollY = m_sceneSettings->GetScrollOffsets().y;

        m_viewMatrix = lookAt(
            m_pos,
            m_target,
            m_up);
    }

    // Update the mouse position for the next rotation
    m_previousCursorPos.x = xpos;
    m_previousCursorPos.y = ypos;

    // For the next frame, the "last time" will be "now"
    lastTime = currentTime;
}

const vec3 &Camera::GetTarget() { return m_target; }

void Camera::SetTarget(const vec3 &target) { 
    m_target = target; 
    m_viewMatrix = lookAt(m_pos, m_target, m_up);
    UpdateWireframe();
    m_forward = normalize(m_pos - m_target);
    m_right = normalize(cross(m_forward, m_up));
    m_realUp = normalize(cross(m_forward, m_right));
    
    m_gizmo->SetX(m_right);
    m_gizmo->SetY(m_realUp);
    m_gizmo->SetZ(m_forward);
    m_gizmo->UpdateLines();

    /** Update center line. */
    WRITE_VEC3(m_centerLineVertices, 3, m_pos + m_forward * m_centerLineLength);
    m_centerLine->UpdateVertices(m_centerLineVertices);
}

const vec3 &Camera::GetRight() { return m_right; }

void Camera::SetRight(const vec3 &v) { 
    m_right = v; 
    UpdateWireframe(); 
    m_gizmo->SetX(normalize(m_right));
    m_gizmo->UpdateLines();
}

const vec3 &Camera::GetRealUp() { return m_realUp; }

const vec3 &Camera::GetUp() { return m_up; }

void Camera::SetUp(const vec3 &v) { 
    m_up = v; 
    UpdateWireframe(); 
    m_gizmo->SetY(normalize(v));
    m_gizmo->UpdateLines();
}

const vec3 &Camera::GetForward() { return m_forward; }

void Camera::SetForward(const vec3 &v) { 
    // m_forward = v; UpdateWireframe();
}

void Camera::Update()
{
    m_forward = normalize(m_pos - m_target);

    m_right = normalize(cross(m_forward, m_up));

    m_realUp = normalize(cross(m_right, m_forward));
}

const float *Camera::GetWireframe()
{
    return m_wireframeVertices;
}

void Camera::UpdateWireframe()
{
    vec3 corner_top_left_tmp = NDCToCamera(vec2(-1.0, 1.0), m_projectionMatrix) * m_wireSize * -1.0f;
    vec3 corner_top_left = CameraToWorld(vec4(corner_top_left_tmp, 1.0f), m_viewMatrix);

    vec3 corner_top_right_tmp = NDCToCamera(vec2(1.0, 1.0), m_projectionMatrix) * m_wireSize * -1.0f;
    vec3 corner_top_right = CameraToWorld(vec4(corner_top_right_tmp, 1.0f), m_viewMatrix);

    vec3 corner_bot_left_tmp = NDCToCamera(vec2(-1.0, -1.0), m_projectionMatrix) * m_wireSize * -1.0f;
    vec3 corner_bot_left = CameraToWorld(vec4(corner_bot_left_tmp, 1.0f), m_viewMatrix);

    vec3 corner_bot_right_tmp = NDCToCamera(vec2(1.0, -1.0), m_projectionMatrix) * m_wireSize * -1.0f;
    vec3 corner_bot_right = CameraToWorld(vec4(corner_bot_right_tmp, 1.0f), m_viewMatrix);

    m_imagePlane->SetVertices(corner_top_left, corner_top_right, corner_bot_left, corner_bot_right);

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

float Camera::GetFovX() { return m_fov.x; }
void Camera::SetFovX(float fov, bool keepRatio)
{
    m_fov.x = fov;
    UpdateWireframe();
    // TODO: ratio.
}

float Camera::GetFovY() { return m_fov.y; }
void Camera::SetFovY(float fov) { m_fov.y = fov; UpdateWireframe();}

float Camera::GetNear() { return m_near; }
void Camera::SetNear(float near) { m_near = near; UpdateWireframe();}

void Camera::SetFar(float far) { m_far = far; UpdateWireframe();}

float Camera::GetFar() { return m_far; }

void Camera::Render()
{
    /** Frustum */
    if(m_showFrustumLines) m_frustumLines->Render();
    m_gizmo->Render();

    /** Image plane linked. */
    if(m_imagePlane != nullptr && m_showImagePlane) m_imagePlane->Render();

    /** Center line. */
    if(m_displayCenterLine){
        m_centerLine->Render();
    }

    /** Rays (partial) for each pixel. */
}

void Camera::SetIntrinsic(const mat4 &intrinsic)
{
    m_projectionMatrix = intrinsic;
    UpdateWireframe();
}

void Camera::SetExtrinsic(const mat4 &extrinsic)
{
    m_viewMatrix = extrinsic;

    mat3 rotMat(m_viewMatrix);
    vec3 d(m_viewMatrix[3]);

    // m_forward = CameraToWorld(vec4(0.0, 0.0, 1.0, 1.0), extrinsic);
    m_forward = -transpose(m_viewMatrix)[2];
    // m_right = CameraToWorld(vec4(1.0, 0.0, 0.0, 1.0), extrinsic);
    m_right = transpose(m_viewMatrix)[0];
    // m_realUp = CameraToWorld(vec4(0.0, 1.0, 0.0, 1.0), extrinsic);
    m_up = vec3(0.0, 1.0, 0.0);
    m_realUp = normalize(cross(m_forward, m_right));

    m_pos = -d * rotMat;

    m_gizmo->SetPosition(m_pos);
    m_gizmo->SetX(m_right * m_wireSize);
    m_gizmo->SetY(m_realUp * m_wireSize);
    m_gizmo->SetZ(m_forward * m_wireSize);
    m_gizmo->UpdateLines();

    UpdateWireframe();

    /** Update center line. */
    WRITE_VEC3(m_centerLineVertices, 0, m_pos);
    WRITE_VEC3(m_centerLineVertices, 3, m_pos + m_forward * m_centerLineLength);
    m_centerLine->UpdateVertices(m_centerLineVertices);
    
}

const mat4 &Camera::GetIntrinsic()
{
    return m_projectionMatrix;
}

const mat4 &Camera::GetExtrinsic()
{
    return m_viewMatrix;
}

const ivec2 &Camera::GetResolution()
{
    return m_resolution;
}

void Camera::SetImage(Image* image){
    if(image == nullptr) return;
    m_imageTex->LoadFromImage(image);
    filename = image->filename;
    m_showImagePlane = true;
}

bool Camera::IsCenterLineVisible(){
    return m_displayCenterLine;
}

void Camera::SetIsCenterLineVisible(bool visible){
    m_displayCenterLine = visible;
}

void Camera::SetCenterLineLength(float length){
    m_centerLineLength = length;

    /** Update center line. */
    WRITE_VEC3(m_centerLineVertices, 3, m_pos + m_forward * m_centerLineLength);
    m_centerLine->UpdateVertices(m_centerLineVertices);
}

float Camera::GetCenterLineLength(){
    return m_centerLineLength;
}

void Camera::SetShowFrustumLines(bool value){
    m_showFrustumLines = value;
}

bool Camera::ShowFrustumLines(){
    return m_showFrustumLines;
}

bool Camera::ShowImagePlane(){
    return m_showImagePlane;
}
void Camera::SetShowImagePlane(bool visible){
    m_showImagePlane = visible;
}