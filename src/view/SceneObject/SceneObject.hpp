#pragma once
#include "../Renderable/Renderable.hpp"
#include <string>
#include <iostream>
#include <vector>

enum SceneObjectTypes
{
    IMAGESET,
    CAMERA,
    LINES,
    LINEGRID,
    VOLUME3D,
    NERFDATASET,
    CAMERASET,
    GIZMO,
    MODEL,
    MESH,
    VOLUMERENDERER,
    NONE
};

static const std::vector<const char *> SceneObjectNames = {
    "ImageSet",
    "Camera",
    "Lines",
    "LineGrid",
    "Volume3D",
    "NeRF Dataset",
    "Camera Set",
    "Gizmo",
    "Model",
    "Mesh",
    "Volume Renderer",
    "None"};

class SceneObject : public Renderable
{
public:
    SceneObject(const std::string typeName, enum SceneObjectTypes type) : m_typeName(typeName), m_type(type), m_id(-1), m_listVisible(true) {}

    /**
     * @brief Get the uniq-id that identify this object.
     *
     * @return int : A uniq-id.
     */
    int GetID() { return m_id; }

    /**
     * @brief Set the uniq-id of this object.
     *
     * @param id : An integer != -1 that uniquely identify this object in the scene.
     */
    void SetID(int id) { m_id = id; }

    /**
     * @brief Checks if the SceneObject is active or not.
     * Corresponds to the checkbox state in the Object list.
     *
     * @return true : If the object is active and have to be visible "rendered".
     * @return false : If the object must be inactive and so not visible.
     */
    bool IsActive() { return m_active; }

    /**
     * @brief Set the active state of this SceneObject.
     *
     * @param active : True to set this object as active and visible. False otherwise.
     */
    void SetActive(bool active) { m_active = active; }

    /**
     * @brief Set this SceneObject's name.
     * Not necessarely unique.
     *
     * @param name : A string that will be displayed as the object's name.
     */
    void SetName(const std::string &name) { m_name = name; }

    /**
     * @brief Get this SceneObject's name.
     *
     * @return const std::string& : The object's name.
     */
    const std::string &GetName() { return m_name; }

    /**
     * @brief Get the object type's name.
     * Identify the Type of the Object as a string.
     *
     * @return const std::string& : A constant string that contains the name of the type used.
     */
    const std::string &GetTypeName() { return m_typeName; }

    /**
     * @brief Get the object type.
     * Identify the Type of the Object.
     * Used to connect with the appropriate inpector in the view.
     *
     * @return const std::string& : A constant string that contains the name of the type used.
     */
    enum SceneObjectTypes GetType() { return m_type; }

    /** Default constructor. Create an invalid SceneObject. */
    SceneObject() : m_id(-1) {}

    std::vector<std::shared_ptr<SceneObject>> &GetChildren() { return m_children; }

    std::shared_ptr<SceneObject> GetChild(int id)
    {
        if (m_children.size() > 0)
        {
            for (auto child : m_children)
            {
                if (child->GetID() == id)
                {
                    return child;
                }
                else
                {
                    auto object_in_child = child->GetChild(id);
                    if (object_in_child != nullptr)
                        return object_in_child;
                }
            }
        }

        return nullptr;
    }

    
    std::vector<std::shared_ptr<SceneObject>> GetAll(SceneObjectTypes type)
    {
        std::vector<std::shared_ptr<SceneObject>> tab = std::vector<std::shared_ptr<SceneObject>>();
        for (auto obj : m_children)
        {
            if (obj->GetType() == type)
                tab.push_back(obj);

            std::vector<std::shared_ptr<SceneObject>> tmp = obj->GetAll(type);
            for (auto a : tmp)
                tab.push_back(a);
        }
        return tab;
    }

    bool IsChild() { return m_isChild; }

    void SetIsChild(bool isChild) { m_isChild = isChild; }

    void SetIsVisibleInList(bool visible) { m_listVisible = visible; }
    bool IsVisibleInList() { return m_listVisible; }

protected:
    bool m_isChild;
    /** A uniq-id that identify this SceneObject. */
    int m_id;
    /** True if the object is active in the scene (visible) or not. */
    bool m_active;
    bool m_listVisible;
    /** The Object's name. */
    std::string m_name;
    /** The Object's type, stored as a string in order to be easily extended. */
    const std::string m_typeName;
    /** The Object's type as the enum, used for fast comparison. */
    enum SceneObjectTypes m_type;

    /** Chilren SceneObjects (dependencies). */
    std::vector<std::shared_ptr<SceneObject>> m_children;
};