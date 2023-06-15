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
#include "../CudaTexture.hpp"

class Scene;
class Lines;
class Gizmo;

using namespace glm;

class Camera : public SceneObject
{
private:
    /** ext dep. */
    Scene * m_scene;
    std::shared_ptr<SceneSettings> m_sceneSettings;

    /** Cursor */
    vec2 m_previousCursorPos = vec2(0.0, 0.0);
    float m_prevScrollY = 0.0f;

    double yDeltaAngle{};

    /** Camera position in world space coordinates. */
    vec3 m_pos{};
    /** The target position in world space that the camera is looking at. */
    vec3 m_target{};
    /** The camera's up vector. */
    vec3 m_up{};
    /** Field of view, (x,y). */
    vec2 m_fov{};

    float m_near = 0.001f;
    float m_far = 100.0f;

    /** Computed values. */
    vec3 m_forward{};
    vec3 m_realUp{};
    vec3 m_right{};

    bool m_displayCenterLine = false;
    bool m_showImagePlane = false;
    bool m_showFrustumLines = true;

    /**
     * @brief Also known as the extrinsic matrix.
     * Rotate and translate compare to world coordinates.
     */
    mat4 m_viewMatrix{};

    /**
     * @brief Also known as the intrinsic matrix.
     * Projection of points from camera-space to image-space.
     */
    mat4 m_projectionMatrix{};

    float m_scroll = 0.0f;
    float m_speed = 3.0f;
    float m_horizontalAngle = 3.14f*1.25f;
    // Initial vertical angle : none
    float m_verticalAngle = -3.14f * 0.2f;
    // Initial Field of View
    float m_initialFoV = 65.0f;

    float m_mouseSpeed = 0.005f;

    float m_wireframeVertices[16*3] = {0.0f};

    ivec2 m_resolution{};

    Lines* m_frustumLines{};
    Gizmo* m_gizmo{};
    // float m_wireSize = 0.1f;
    float m_wireSize = 0.15f;

    /** Image plane. */
    Texture2D* m_imageTex = nullptr;
    Plane* m_imagePlane = nullptr;
    /** Center line forward dir. */
    Lines* m_centerLine{};
    float m_centerLineVertices[6] = {0.0f};
    float m_centerLineLength = 1.0f;

    GPUData<CameraDescriptor> m_desc;
    GPUData<IntegrationRangeDescriptor> m_integrationDesc;

    std::shared_ptr<CudaTexture> m_cudaTexture = nullptr;

public:
    Camera(Scene *scene, const std::string& name, const vec3& position, const vec3& target);
    explicit Camera(Scene* scene);

    ~Camera() override;

    void Initialize();

    /**
     * Get a descriptor of the camera available on GPU memory.
     * @return : A GPU-Ready pointer to a Camera Descriptor.
     */
    CameraDescriptor* GetGPUDescriptor();

    /**
     * Update the GPU Descriptor on GPU memory.
     */
    void UpdateGPUDescriptor();

    GPUData<CameraDescriptor>& GetGPUData();
    GPUData<IntegrationRangeDescriptor>& GetIntegrationRangeGPUDescriptor();

    std::shared_ptr<CudaTexture> GetCudaTexture();

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
     * @brief Get the camera's up vector in world space coordinates.
     * 
     * @return vec3 : Up vector.
     */
    const vec3& GetUp();
    void SetUp(const vec3&);

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
     * @brief Get the View Matrix also known as the extrinsic matrix.
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
     * //TODO: Move in the view. (Detach dependencies).
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
    [[nodiscard]] float GetFovX() const;
    
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
    [[nodiscard]] float GetFovY() const;

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
    [[nodiscard]] float GetNear() const;

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
    [[nodiscard]] float GetFar() const;

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
     * @brief Get the Resolution of the camera in pixels. 
     * 
     * @return const ivec2& (x = width, y = height), amount of pixels in both axis.
     */
    const ivec2& GetResolution();

    ivec2 SetResolution(const ivec2& res);

    /**
     * @brief Render the camera in the scene. Render the gizmo, frustum and potentially 
     * image planes of the camera's images is activated. 
     */
    void Render() override;

    void InitializeCudaTexture();

    /**
     * @brief Set the Image displayed on the current image plane.
     * 
     * @param image : A pointer to the image to display.
     */
    void SetImage(Image* image);

    [[nodiscard]] bool IsCenterLineVisible() const;

    void SetIsCenterLineVisible(bool visible);

    void SetCenterLineLength(float length);

    [[nodiscard]] float GetCenterLineLength() const;

    void SetShowFrustumLines(bool value);
    [[nodiscard]] bool ShowFrustumLines() const;

    [[nodiscard]] bool ShowImagePlane() const;
    void SetShowImagePlane(bool visible);

    void SetFrustumSize(float value);

    /**
     * @brief Filename related to the source image of the camera.
     * 
     */
    std::string filename;


};