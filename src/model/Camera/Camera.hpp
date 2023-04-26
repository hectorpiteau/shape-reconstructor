#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "../utils/Utils.hpp"
#include "../utils/SceneSettings.hpp"
#include "../view/Lines.hpp"
#include "../view/Gizmo.hpp"
#include "../view/SceneObject/SceneObject.hpp"
#include "../controllers/Scene/Scene.hpp"
#include "../Texture2D.hpp"
#include "../view/Plane.hpp"

class Scene;
class Lines;
class Gizmo;

using namespace glm;

class Camera : public SceneObject
{

public:
    Camera(Scene *scene, const std::string& name, const vec3& position, const vec3& target);
    Camera(Scene* scene);
    ~Camera();

    void Initialize();

    /**
     * @brief Set the camera's position in world space coordinates.
     * 
     * @param position : The camera position in world space coordinates.
     */
    void SetPosition(const vec3& position);

    /**
     * @brief Set the camera's position in world space coordinates.
     * 
     * @param x : The x coordinate in world space coordinates.
     * @param y : The y coordinate in world space coordinates.
     * @param z : The z coordinate in world space coordinates.
     */
    void SetPosition(float x, float y, float z);
    
    /**
     * @brief Get the camera's position in world space.
     * 
     * @return vec3 : World position.
     */
    const vec3& GetPosition();
    
    /**
     * @brief Get the camera's right vector, in world space coordinates.
     * 
     * @return vec3 : Right vector.
     */
    const vec3& GetRight();
    void SetRight(const vec3& v);

    /**
     * @brief Get the camera's real up vector, (the one orthogonal to 
     * forward and right vectors), in world space coordinates. 
     * 
     * @return vec3 : Real up vector.
     */
    const vec3& GetRealUp();

    /**
     * @brief Get the camera's up vector in world space coordinates.
     * 
     * @return vec3 : Up vector.
     */
    const vec3& GetUp();
    void SetUp(const vec3&);

    /**
     * @brief Get the camera's forward vector in world space coordinates.
     * 
     * @return vec3 : Forward vector.
     */
    const vec3& GetForward();
    void SetForward(const vec3& v);

    /**
     * @brief Get the camera's target (lookAt) in world space coordinates.
     * Where the camera is looking at.
     * 
     * @return vec3 : The camera lookAt point in world space coordinates.
     */
    const vec3& GetTarget();
    /**
     * @brief Set the target's of the camera. The point that the camera
     * is looking at.
     * 
     * @param target : A coordinate in world space coordinates.
     */
    void SetTarget(const vec3& target);
    
    /**
     * @brief Computes the View / Projection (extrinsic, intrinsic) matrices 
     * from the mouse and keyboard inputs.
     */
    void ComputeMatricesFromInputs(GLFWwindow *window);

    /**
     * @brief Get the View Matrix also known as the extrinsics matrix.
     * 
     * @return const mat4& : A constant reference to the matrix.
     */
    const mat4& GetViewMatrix();

    /**
     * @brief Get the Projection Matrix also known as the intrinsics matrix.
     * 
     * @return const mat4& 
     */
    const mat4& GetProjectionMatrix();

    /**
     * @brief Get the camera's Wireframe representation.
     * //TODO: Move in the view. (Detach dependencie). 
     * 
     * @return const float* : A constant list of floats that represents points in space that defines
     * a list of lines (wireframe).
     */
    const float* GetWireframe();
    
    /**
     * @brief Updates the wireframe with the current camera's parameters.
     * 
     */
    void UpdateWireframe();

    /**
     * @brief Set the Field of View horizontal direction of the camera.
     * 
     * @param fov : Angle in radian.
     * @param keepRatio : True if the ratio height/width 
     */
    void SetFovX(float fov, bool keepRatio = false);
    
    /**
     * @brief Get the Field of View horizontal direction.
     * 
     * @return float : The angle in radian. 
     */
    float GetFovX();
    
    /**
     * @brief Set the Field of View vertical direction of the camera.
     * 
     * @param fov : Angle in radian.
     */
    void SetFovY(float fov);

    /**
     * @brief Get the Field of View vertical direction.
     * 
     * @return float : The angle in radian.
     */
    float GetFovY();

    /**
     * @brief Set the Near clip plane of the camera.
     * 
     * @param near : Near distance. 
     */
    void SetNear(float near);

    /**
     * @brief Get the Near clip plane distance.
     * 
     * @return float : Near clip plane distance.
     */
    float GetNear();

    /**
     * @brief Set the Far clip plane distance.
     * 
     * @param far : Far clip plane distance.
     */
    void SetFar(float far);

    /**
     * @brief Get the Far clip plane distance.
     * 
     * @return float : The far clip plane distance.
     */
    float GetFar();

    /**
     * @brief Set the camera's intrinsic matrix.
     * 
     * @param intrinsic : K matrix (field of view, principal point, skew).
     */
    void SetIntrinsic(const mat4& intrinsic);
    /**
     * @brief Set the camera's extrinsic matrix.
     * 
     * @param extrinsic : Position and orientation with respect to the world.
     */
    void SetExtrinsic(const mat4& extrinsic);
    
    /**
     * @brief Get the camera's intrinsic matrix.
     * 
     * @return const mat4& : A ref to the intrinsic matrix.
     */
    const mat4& GetIntrinsic();

    /**
     * @brief Get the camera's extrinsic matrix. 
     * 
     * @return const mat4& A ref to the the camera's extrinsic matrix.
     */
    const mat4& GetExtrinsic();
    
    /**
     * @brief TODO
     * 
     */
    void Update();
    
    /**
     * @brief Get the Resolution of the camera in pixels. 
     * 
     * @return const vec2& (x = width, y = height), amount of pixels in both axis.
     */
    const vec2& GetResolution();

    /**
     * @brief Render the camera in the scene. Render the gizmo, frustum and potentially 
     * image planes of the camera's images is activated. 
     */
    void Render();

    /**
     * @brief Set the Image dislpayed on the current image plane.
     * 
     * @param image : A pointer to the image to display.
     */
    void SetImage(Image* image);

    bool IsCenterLineVisible();

    void SetIsCenterLineVisible(bool visible);

    void SetCenterLineLength(float length);

    float GetCenterLineLength();

    void SetShowFrustumLines(bool value);
    bool ShowFrustumLines();

    bool ShowImagePlane();
    void SetShowImagePlane(bool visible);

    /**
     * @brief Filename releated to the source image of the camera. 
     * 
     */
    std::string filename;

private:
    /** ext dep. */
    Scene * m_scene;
    std::shared_ptr<SceneSettings> m_sceneSettings;

    /** Cursor */
    vec2 m_previousCursorPos;
    float m_prevScrollY = 0.0f;
    
    double yDeltaAngle;

    /** Camera position in world space coordinates. */
    vec3 m_pos;
    /** The target position in world space that the camera is looking at. */
    vec3 m_target;
    /** The camera's up vector. */
    vec3 m_up;
    /** Field of view, (x,y). */
    vec2 m_fov;

    float m_near = 0.001f;
    float m_far = 100.0f;
    
    /** Computed values. */
    vec3 m_forward;
    vec3 m_realUp;
    vec3 m_right;

    bool m_displayCenterLine = false;
    bool m_showImagePlane = false;
    bool m_showFrustumLines = true;
    
    /**
     * @brief Also known as the extrinsic matrix.
     * Rotate and translate compare to world coordinates.
     */
    mat4 m_viewMatrix;

    /**
     * @brief Also known as the intrinsic matrix.
     * Projection of points from camera-space to image-space.
     */
    mat4 m_projectionMatrix;

    float m_scroll = 0.0f;
    float m_speed = 3.0f;
    float m_horizontalAngle = 3.14f*1.25f;
    // Initial vertical angle : none
    float m_verticalAngle = -3.14f * 0.2f;
    // Initial Field of View
    float m_initialFoV = 65.0f;

    float m_mouseSpeed = 0.005f;

    float m_wireframeVertices[16*3] = {0.0f};

    vec2 m_resolution;

    Lines* m_frustumLines;
    Gizmo* m_gizmo;
    // float m_wireSize = 0.1f;
    float m_wireSize = 1.0f;

    /** Image plane. */
    Texture2D* m_imageTex = nullptr;
    Plane* m_imagePlane = nullptr;
    /** Center line forward dir. */
    Lines* m_centerLine;
    float m_centerLineVertices[6] = {0.0f};
    
    float m_centerLineLength = 1.0f;

};