#pragma once

#include <vector>
#include <string>

#include "../model/Image.hpp"
#include "SceneObject/SceneObject.hpp"

class ImageSet : public SceneObject
{
public:
    ImageSet();

    ImageSet(const ImageSet &) = delete;

    void SetFolderPath(const std::string &path);

    void LoadImages();

    int GetAmountOfImages();

    const Image *GetImage(int index);

    void Render(const glm::mat4 &projection, const glm::mat4 &view, std::shared_ptr<SceneSettings> scene);

private:
    std::vector<Image *> m_images;

    std::string m_folderPath;
};