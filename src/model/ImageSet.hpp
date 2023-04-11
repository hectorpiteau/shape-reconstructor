#pragma once

#include <vector>
#include <string>

#include "Image.hpp"
#include "../view/SceneObject/SceneObject.hpp"

class ImageSet : public SceneObject
{
public:
    ImageSet();

    ImageSet(const ImageSet &) = delete;

    void SetFolderPath(const std::string &path);

    /**
     * @brief Load the images contained in the folder path.
     * 
     * @return size_t : The amount of images loaded. 
     */
    size_t LoadImages();

    int GetAmountOfImages();

    const Image *GetImage(int index);
    
    std::vector<Image *>& GetImages();
    const Image* operator[](size_t index);

    size_t size();
    
    void Render(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene);

private:
    std::vector<Image *> m_images;

    std::string m_folderPath;
};