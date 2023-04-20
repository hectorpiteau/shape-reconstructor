#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <memory>
#include <iostream>
#include "../../include/stb_image.h"
#include "../utils/Utils.hpp"
#include "Image.hpp"

class Texture2D
{
private:
    int m_width;
    int m_height;
    int m_channels;
    short m_bytesPerItem;
    GLenum m_textureID;
    GLuint m_ID;

    /** Data owned by this class, responsible for allocation and deletion. */
    unsigned char *m_data;
    /** Data length in byte. */
    size_t m_dataLength;

    /** Data from outside, not responsible for allocation and deletion. */
    const unsigned char *m_refData;
    bool m_useRefData;

    void LoadFromData(unsigned char *data)
    {
        if (data != nullptr)
        {
            if (m_useRefData) m_refData = data;
            else m_data = data;
            
            std::cout <<"Load texture from data: channels :" << std::to_string(m_channels) << std::endl;
            
            GLenum format;
            
            if (m_channels == 1)
                format = GL_RED;
            else if (m_channels == 3)
                format = GL_RGB;
            else if (m_channels == 4)
                format = GL_RGBA;

            glGenTextures(1, &m_ID);
            glBindTexture(GL_TEXTURE_2D, m_ID);

            // Setup filtering parameters for display
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            glTexImage2D(GL_TEXTURE_2D, 0, format, m_width, m_height, 0, format, GL_UNSIGNED_BYTE, m_useRefData ? m_refData : m_data);
            
            if(m_useRefData == false) stbi_image_free(m_data);
        }
    }

public:

    Texture2D(int width, int height, int channels, short bytesPerItems) : m_width(width), m_height(height), m_channels(channels), m_bytesPerItem(bytesPerItems)
    {
        m_data = nullptr;
        m_refData = nullptr;
        m_useRefData = false;

        m_dataLength = m_width * m_height * m_channels * m_bytesPerItem;

        glGenTextures(1, &m_textureID);
        glBindTexture(GL_TEXTURE_2D, m_textureID);

        m_data = (unsigned char *)malloc(sizeof(unsigned char) * m_dataLength);
        if (m_data == nullptr)
        {
            std::cerr << "Texture2D : Error allocating memory for data of length: " << m_dataLength << " bytes." << std::endl;
            return;
        }

        InitData();

        UpdateMemData();
    };

    /**
     * @brief Construct a new Texture 2D from an image file.
     *
     * @param path
     */
    Texture2D(const std::string &path)
    {
        stbi_set_flip_vertically_on_load(true);
        m_data = nullptr;
        m_refData = nullptr;
        m_useRefData = false;
        unsigned char *data = stbi_load(path.c_str(), &m_width, &m_height, &m_channels, 0);
        

        LoadFromData(data);
    }

    Texture2D()
    {
        m_data = nullptr;
        m_refData = nullptr;
        m_useRefData = false;
    }

    /**
     * @brief Load the texture from an Image instance.
     * 
     * @param image 
     */
    void LoadFromImage(const Image *image)
    {
        m_width = image->width;
        m_height = image->height;
        m_channels = image->channels;
        m_useRefData = true;
        LoadFromData(image->data);
    }

    void InitData()
    {
        memset(m_data, 0, m_dataLength);

        float centerX = ((float)m_width) / 2.0f;
        float centerY = ((float)m_height) / 2.0f;

        for (int i = 0; i < m_height; i += 1)
        {
            for (int j = 0; j < m_width; j += 1)
            {
                float res = sqrt((((float)j) - centerX) * (((float)j) - centerX) + (((float)i) - centerY) * (((float)i) - centerY));

                m_data[i * m_channels * m_width + j * m_channels] = (char)res;
                m_data[i * m_channels * m_width + j * m_channels + 1] = (char)res;
                m_data[i * m_channels * m_width + j * m_channels + 2] = (char)res;
                m_data[i * m_channels * m_width + j * m_channels + 3] = (char)255;
            }
        }
    }

    void UpdateMemData()
    {
        // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_useRefData ? m_refData : m_data);
        // glGenerateMipmap(GL_TEXTURE_2D);
    }

    bool SetDataByCopy(const char *data, size_t length)
    {
        if (length > m_dataLength)
        {
            std::cerr << "Texture2D: SetDataByCopy, input data is too big for this texture. Need to reallocate." << std::endl;
            return false;
        }

        m_useRefData = false;
        memcpy(m_data, data, length);

        return true;
    }

    void SetDataAsReference(const unsigned char *data)
    {
        m_useRefData = true;
        m_refData = data;
    }

    GLenum GetID() { return m_ID; }

    void BindTexture2D(){
        glBindTexture(GL_TEXTURE_2D, m_ID);
    }

    int GetWidth() { return m_width; }
    int GetHeight() { return m_height; }

    ~Texture2D()
    {
        if (m_data != nullptr)
            delete[] m_data;
    };
};