#ifndef TEXTURE2D_H
#define TEXTURE2D_H

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <memory>
#include <iostream>
#include "Utils.hpp"

class Texture2D
{
private:
    int m_width;
    int m_height;
    int m_channels;
    short m_bytesPerItem;
    GLenum m_textureID;
    

    /** Data owned by this class, responsible for allocation and deletion. */
    char *m_data;
    /** Data length in byte. */
    size_t m_dataLength;

    /** Data from outside, not responsible for allocation and deletion. */
    const char *m_refData;
    bool m_useRefData;

public:
    Texture2D(int width, int height, int channels, short bytesPerItems) : m_width(width), m_height(height), m_channels(channels), m_bytesPerItem(bytesPerItems)
    {
        m_data = nullptr;
        m_refData = nullptr;
        m_useRefData = false;

        m_dataLength = m_width * m_height * m_channels * m_bytesPerItem;

        glGenTextures(1, &m_textureID);
        glBindTexture(GL_TEXTURE_2D, m_textureID);

        m_data = (char*) malloc(sizeof(char) * m_dataLength);
        if(m_data == nullptr){
            std::cerr << "Texture2D : Error allocating memory for data of length: " << m_dataLength << " bytes." << std::endl;
            return;
        }

        InitData();

        UpdateMemData();
    };

    void InitData(){
        memset(m_data, 0, m_dataLength);

        float centerX = ((float)m_width)/2.0f;
        float centerY = ((float)m_height)/2.0f;
        

        for(int i=0; i < m_height; i += 1){
            for(int j=0; j < m_width; j += 1){
                float res = sqrt(( ((float)j)-centerX)*(((float)j)-centerX) + (((float)i)-centerY) * (((float)i)-centerY));
                
                m_data[i* m_channels * m_width + j*m_channels] = (char)res;
                m_data[i* m_channels * m_width + j*m_channels + 1] = (char)res;
                m_data[i* m_channels * m_width + j*m_channels + 2] = (char)res;
                m_data[i* m_channels * m_width + j*m_channels + 3] = (char)255;
            }
        }
    }

    void UpdateMemData()
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_useRefData ? m_refData : m_data);
        glGenerateMipmap(GL_TEXTURE_2D);
        
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

    void SetDataAsReference(const char *data)
    {
        m_useRefData = true;
        m_refData = data;
    }

    GLenum GetID() { return m_textureID; }

    ~Texture2D()
    {
        if (m_data != nullptr)
            delete[] m_data;
    };
};

#endif // TEXTURE2D_H