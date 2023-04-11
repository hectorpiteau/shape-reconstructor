#pragma once 
#include "../Renderable/Renderable.hpp"
#include <string>
#include <iostream>

enum SceneObjectTypes {
    IMAGESET,
    CAMERA,
    LINES,
    LINEGRID,
    VOLUME3D,
    NONE
};

class SceneObject : public Renderable{
public:
    
    SceneObject(const std::string typeName, enum SceneObjectTypes type) : m_typeName(typeName), m_type(type), m_id(-1){}

    /**
     * @brief Get the uniq-id that identify this object.
     * 
     * @return int : A uniq-id.
     */
    int GetID(){ return m_id;}
    
    /**
     * @brief Set the uniq-id of this object.
     * 
     * @param id : An integer != -1 that uniquely identify this object in the scene.
     */
    void SetID(int id) { m_id = id;}

    /**
     * @brief Checks if the SceneObject is active or not. 
     * Corresponds to the checkbox state in the Object list.
     * 
     * @return true : If the object is active and have to be visible "rendered".
     * @return false : If the object must be inactive and so not visible.
     */
    bool IsActive(){ return m_active;}
    
    /**
     * @brief Set the active state of this SceneObject.
     * 
     * @param active : True to set this object as active and visible. False otherwise.
     */
    void SetActive(bool active) { m_active = active; std::cout << "SceneObject SetActive: " << active << std::endl; }

    /**
     * @brief Set this SceneObject's name.
     * Not necessarely unique.
     * 
     * @param name : A string that will be displayed as the object's name.
     */
    void SetName(const std::string &name){ m_name = name;}

    /**
     * @brief Get this SceneObject's name.
     * 
     * @return const std::string& : The object's name.
     */
    const std::string& GetName(){ return m_name;}

    /**
     * @brief Get the object type's name.
     * Identify the Type of the Object as a string.
     * 
     * @return const std::string& : A constant string that contains the name of the type used. 
     */
    const std::string& GetTypeName(){ return m_typeName; }

    /**
     * @brief Get the object type.
     * Identify the Type of the Object.
     * Used to connect with the appropriate inpector in the view.
     * 
     * @return const std::string& : A constant string that contains the name of the type used. 
     */
    enum SceneObjectTypes GetType(){ return m_type; }

    /** Remove copy constructor. */
    SceneObject(const SceneObject&) = delete;
    
    /** Default constructor. Create an invalid SceneObject. */
    SceneObject() : m_id(-1){}

protected:
    /** A uniq-id that identify this SceneObject. */
    int m_id;
    /** True if the object is active in the scene (visible) or not. */
    bool m_active;
    /** The Object's name. */
    std::string m_name;
    /** The Object's type, stored as a string in order to be easily extended. */
    const std::string m_typeName;
    /** The Object's type as the enum, used for fast comparison. */
    enum SceneObjectTypes m_type;
};