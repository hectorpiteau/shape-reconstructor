#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <iostream>
#include <glm/glm.hpp>
#include "../../include/stb_image.h"

using namespace glm;

struct Patch {
    /** The patch origin on the original image. 
     * Unit is in pixel. Origin is the top-left 
     * corner of the image. 
     * Total: 16+16=32bits 
     * */
    unsigned short x, y;

    /** Pointer to data. */
    unsigned char* data;
};

struct Patch {
    /** The patch origin on the original image. 
     * Unit is in pixel. Origin is the top-left 
     * corner of the image. 
     * Total: 16+16=32bits 
     * */
    unsigned short x, y;

    /** Pointer to data. */
    unsigned char* data;
};

class Image
{
public:
    Image(std::string filename): filename(filename) {};
    ~Image() {};
    int width;
    int height;
    int channels;
    int bits;
    std::string filename;

    /** row-major storage */
    unsigned char* data;

    int GetWidth(){ return width;}
    int GetHeight(){ return height;}
    int GetChannels(){ return channels;}
    const std::string& GetFilename(){ return filename;}
    
    size_t GetImageMemorySize(){ return width * height * channels * sizeof(unsigned char);}


    void LoadPng(const std::string& path, bool verticalFlip, bool linearF) {
        stbi_set_flip_vertically_on_load(verticalFlip);

        if(!linearF) { //no float data
            data = stbi_load(path.c_str(), &width, &height, &channels, 0);
        } else { //float data like hdr images
            data = reinterpret_cast<unsigned char*>(stbi_loadf(path.c_str(), &width, &height, &channels, 0));
            bits = 8 * sizeof(float);
        }

        if(data == nullptr) {
            std::cout << "[Image] could not load png image: {}" << path << std::endl;
            std::cout << "[Image] error: {}" << stbi_failure_reason() << std::endl;
            m_isLoaded = false;
            return;
        }
        m_isLoaded = true;
    }

    /**
     * @brief 
     * 
     * @param isBackgroundTransparent 
     * @param patchSize 
     */
    void Patchify(unsigned short patchWidth, unsigned short patchHeight, bool isBackgroundTransparent = true){
        m_patchWidth = patchWidth;
        m_patchHeight = patchHeight;

        int alpha_min = 1;

        // #pragma omp parallel for collapse(2)
        for(unsigned short y=0; y < height; y += m_patchHeight){
            for(unsigned short x=0; x < width; x += m_patchWidth){
                /** Search for valid pixels in the patch. */
                /** This process can be optimized. omp or cuda kernel. */
                for(unsigned short i=0; i<m_patchHeight; i++){
                    for(unsigned short j=0; j<m_patchWidth; j++){
                        /** Get the (y+i, x+j) pixel and compare the 4th value (the alpha) with
                         * the min-alpha. */
                        if(data[((y+i) * width + (x+j)) * 4 + 3 ] > alpha_min){
                            /** Create a patch. */
                            Patch patch = {
                                .x = (unsigned short)(x+j),
                                .y = (unsigned short)(y+i),
                                .data = &data[((y+i) * width + (x+j)) * 4]
                            };

                            /** Critical operation that cannot be parallelized. Must use locks with openmp. */
                            m_patches.push_back(patch);
                        }
                    }
                }
            }
        }
        m_patchesLoaded = true;
    };


    /** Upload patchs on the GPU.*/
    void UploadPatchs(){

    }

    void UnloadImage(){
        free(data);
        m_isLoaded = false;
    }

    void UnloadPatches(){

    }

    const std::string& GetFilename() const {return filename;}

    bool IsLoaded(){return m_isLoaded;}
private:
    bool m_isLoaded = false;
    bool m_patchesLoaded = false;

    std::vector<Patch> m_patches;
    unsigned short m_patchWidth;
    unsigned short m_patchHeight;
};

#endif // IMAGE_H