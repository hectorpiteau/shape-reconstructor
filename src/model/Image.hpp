#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <iostream>
#include <glm/glm.hpp>
#include <utility>
#include "../../include/stb_image.h"
#include "Common.cuh"
#include "GPUData.cuh"

using namespace glm;

//struct Patch {
//    /** The patch origin on the original image.
//     * Unit is in pixel. Origin is the top-left
//     * corner of the image.
//     * Total: 16+16=32bits
//     * ushort = max 65535 values. (safe enough)
//     * */
//    unsigned short x, y;
//
//    /** Pointer to data. */
//    unsigned char *data;
//
//    /** True if the patch is fully on the image, false if it is on a border and overlap void.*/
//    bool isComplete;
//};

struct Patch {
    /** The patch origin on the original image.
     * Unit is in pixel. Origin is the top-left
     * corner of the image.
     * Total: 16+16=32bits
     * */
    unsigned short x, y;

    /** 
     * Image data. column-major indexing. Smallest increment is in the y-axis. 
     * (y, x, RGBA) => 4 x 4 x (4 components).
    */
    unsigned char data[4 * 4 * 4];
};

class Image
{
private:
    bool m_isLoaded = false;
    bool m_isOnGpu = false;

    /** ********** Patch related information ********** */
    short m_patchHeight = 8;
    short m_patchWidth = 8;
    std::vector<Patch> m_patches;
    /** ********** ************************* ********** */

    GPUData<LinearImageDescriptor> m_desc;

public:
    explicit Image(std::string filename, int width = 100, int height = 100) :
            width(width),
            height(height),
            channels(4),
            bits(8 * sizeof(float)),
            filename(std::move(filename)),
            data(nullptr) {
        m_desc.Host()->res = ivec2(width, height);
        m_desc.Host()->data = data;
    };

    ~Image() = default;

    int width;
    int height;
    int channels;
    int bits;
    std::string filename;

    /** row-major storage */
    unsigned char *data = nullptr;

    void LoadGPUData(){
        if(data == nullptr){
            std::cerr << "Image: Cannot generate GPU Data, image is not loaded." << std::endl;
            return;
        }
        size_t size = GetImageMemorySize();

        unsigned char* ptr;
        checkCudaErrors(cudaMalloc((void **) &ptr, size));
        m_desc.Host()->data = ptr;

        std::cout << "Image: Allocate cudaMemory: " << filename << std::endl;

        checkCudaErrors(
                cudaMemcpy((void *) m_desc.Host()->data, (void *) data, size, cudaMemcpyHostToDevice)
        );

        m_desc.Host()->res = ivec2(width, height);
        m_desc.ToDevice();

        m_isOnGpu = true;
    }

    /**
     * Get the GPU Linear Descriptor of this image.
     * Allocates memory on GPU and copy all data on GPU ready to be processed.
     *
     * @return A Device-Ready pointer to a struct containing the image.
     */
    LinearImageDescriptor* GetGPUDescriptor() {
        if(data == nullptr){
            std::cerr << "Image: Cannot generate GPU Data, image is not loaded." << std::endl;
            return nullptr;
        }
        return m_desc.Device();
    }

    /**
     * Free the buffer used for storing the image's data on GPU.
     */
    void FreeGPUDescriptor() {
        std::cout << "Image: Free cudaMemory: " << filename << ", " << std::to_string(m_isOnGpu) << std::endl;
        if(!m_isOnGpu) return;
        /** Free the linear buffer allocated on GPU. */
        GPUData<LinearImageDescriptor>::FreeOnDevice(m_desc.Host()->data);
        m_isOnGpu = false;
    }

    [[nodiscard]] int GetWidth() const { return width; }

    [[nodiscard]] int GetHeight() const { return height; }

    [[nodiscard]] int GetChannels() const { return channels; }

    [[nodiscard]] size_t GetImageMemorySize() const { return width * height * channels * sizeof(unsigned char); }


    void LoadPng(const std::string &path, bool verticalFlip, bool linearF) {
        stbi_set_flip_vertically_on_load(verticalFlip);

        if (!linearF) { //no float data
            data = stbi_load(path.c_str(), &width, &height, &channels, 0);
        } else { //float data like hdr images
            data = reinterpret_cast<unsigned char *>(stbi_loadf(path.c_str(), &width, &height, &channels, 0));
        }

        if (data == nullptr) {
            std::cout << "[Image] could not load png image: {}" << path << std::endl;
            std::cout << "[Image] error: {}" << stbi_failure_reason() << std::endl;
            m_isLoaded = false;
            return;
        }
        std::cout << "[Image] image loaded: (" << std::to_string(width) << ", " << std::to_string(height) << ")" << std::endl;

        m_isLoaded = true;
    }

    /**
     * @brief Initialize the set of patch descriptors for this image.
     * It will not duplicate the image data. Only descriptors with pointers
     * to the main image buffer will be created and stored.
     * 
     * Note: It will only create patches on the section that contains an alpha 
     * value > 1 (on a scale of 0-255).
     * 
     * @param patchWidth : The width of one patch. (default: 8)
     * @param patchHeight : The height of one patch. (default: 8)
     */
   void Patchify(unsigned short patchWidth = 8, unsigned short patchHeight = 8) {
       m_patchWidth = patchWidth;
       m_patchHeight = patchHeight;

       short alpha_min = 1;

       // #pragma omp parallel for collapse(2)
       for (unsigned short y = 0; y < height; y += m_patchHeight) {
           for (unsigned short x = 0; x < width; x += m_patchWidth) {
               /** Search for valid pixels in the patch. */
               /** This process can be optimized. omp or cuda kernel. */

               bool is_patch_valid = false;

               for (unsigned short i = 0; i < m_patchHeight; i++) {
                   for (unsigned short j = 0; j < m_patchWidth; j++) {
                       /** Get the (y+i, x+j) pixel and compare the 4th value (the alpha) with
                        * the min-alpha. */
                       if (data[((y + i) * width + (x + j)) * 4 + 3] > alpha_min) {
                           is_patch_valid = true;
                           break;
                       }
                   }
                   if (is_patch_valid) break;
               }

               if (is_patch_valid) {
                   /** create and store the new patch. */
                   Patch patch = {
                           .x = (unsigned short) (x),
                           .y = (unsigned short) (y)
//                           .data = &data[((y) * width + (x)) * 4],
//                           .isComplete = (x + m_patchWidth < width && y + patchHeight < height) ? true : false
                   };

                   /** Critical operation that cannot be parallelized. Must use locks with openmp. */
                   m_patches.push_back(patch);
               }
           }
       }
//       m_patchesLoaded = true;
   };

    void UnloadImage() {
        free(data);
        m_isLoaded = false;
    }

//    void UnloadPatches() {
//        m_patches = std::vector<Patch>();
//        m_patchesLoaded = false;
//    }

    /**
     * @brief If it exist, get the image's filename.
     * 
     * @return const std::string& : A constant ref to the image's filename.
     */
    [[nodiscard]] const std::string &GetFilename() const { return filename; }

    /**
     * @brief Check if the image is loaded in ram memory or not.
     * 
     * @return true : The image is loaded 
     * @return false : The image is still on disk (no no not loaded).
     */
    [[nodiscard]] bool IsLoaded() const { return m_isLoaded; }
};

#endif // IMAGE_H