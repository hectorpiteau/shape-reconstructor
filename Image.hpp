#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <iostream>
#include <stb_image.h>

class Image
{
public:
    Image() {};
    ~Image() {};
    int width;
    int height;
    int channels;
    int bits;

    unsigned char* data;

    void LoadPng(const std::string& path, bool verticalFlip, bool linearF) {
        stbi_set_flip_vertically_on_load(verticalFlip);

        if(!linearF) { //no float data
            data = stbi_load(path.c_str(), &width, &height, &channels, 0);
        } else { //float data like hdr images
            data = reinterpret_cast<const unsigned char*>(stbi_loadf(path.c_str(), &width, &height, &channels, 0));
            bits = 8 * sizeof(float);
        }

        if(data == nullptr) {
            std::cout << "[Image] could not load png image: {}" << path << std::endl;
            std::cout << "[Image] error: {}" << stbi_failure_reason() << std::endl;
            return;
        }
    }
};

#endif // IMAGE_H