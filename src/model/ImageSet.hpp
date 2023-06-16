#pragma once

#include <vector>
#include <string>
#include <GL/glew.h>
#include "Image.hpp"
#include "../view/SceneObject/SceneObject.hpp"
#include "../controllers/Scene/Scene.hpp"

class ImageSet : public SceneObject
{
public:
    explicit ImageSet(Scene *scene);

    ImageSet(const ImageSet &) = delete;

    void SetFolderPath(const std::string &path);

    const std::string &GetFolderPath();

    /**
     * @brief Load the images contained in the folder path.
     *
     * @return size_t : The amount of images loaded.
     */
    size_t LoadImages();



    Image *GetImage(size_t index);
    Image *GetImage(const std::string &filename);

    std::vector<Image *> &GetImages();
    const Image *operator[](size_t index);

    /**
     * @brief Get the Amount Of Images in this imageSet.
     * If the images are not loaded, the amount is 0.
     *
     * @return size_t : The amount of loaded images.
     */
    size_t size();

    void Render() override;

    bool AreImagesGenerated() const;

private:
    std::vector<Image *> m_images;

    std::string m_folderPath;

    bool m_areImagesGenerated = false;
};